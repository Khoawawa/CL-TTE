import copy
import time
from typing import Dict
import gc

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metric import calculate_metrics
from utils.util import save_model, to_var

from models.loss.loss_balancer import LossBalancer

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag
        
def train_model(model: nn.Module, data_loaders: Dict[str, DataLoader],
                loss_func: callable, optimizer: torch.optim,
                model_folder: str, args, start_epoch=-1, **kwargs):
    loss_balancer = LossBalancer()
    num_epochs = args.epochs
    phases = [
        'train',
        'val',
        ]
    
    since = time.perf_counter()
    
    with open(model_folder + "/output.txt", "a") as f:
        f.write(str(model))
        f.write("\n\n")

    save_dict, best_mae = {'state_dict': copy.deepcopy(model.state_dict()),
                           'epoch': 0
                           }, 10000    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.2, patience=2,
                                                     threshold=1e-4, threshold_mode='rel', min_lr=1e-7)
    if hasattr(args, 'scheduler_state_dict'):
        scheduler.load_state_dict(args.scheduler_state_dict)
    print("LR: ", optimizer.param_groups[0]['lr'])
    scaler = torch.amp.GradScaler()
    try:
        for epoch in range(start_epoch + 1, num_epochs):
            running_loss = {phase: 0.0 for phase in phases}
            msg = []
            for phase in phases:
                args.phase = phase
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                    
                steps, predictions, targets = 0, list(), list()
                
                tqdm_loader = tqdm(data_loaders[phase],mininterval=3)
                for features, truth_data in tqdm_loader:
                    steps += truth_data.size(0)
                    
                    features = to_var(features, args.device)
                    truth_data = to_var(truth_data, args.device)
                    truth_data = torch.clamp(truth_data, min=0.0)
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.amp.autocast(args.device):
                            output, loss_cl = model(features,truth_data)       
                                 
                            loss_eta = loss_func(truth=truth_data, predict=output)
                            
                            if phase == 'train':  
                                loss = loss_balancer(loss_eta, loss_cl, args.beta)
                            else:
                                loss = loss_eta
                        
                        if phase == 'train':    
                            optimizer.zero_grad()
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                    if phase == 'train':       
                        desc = f"L1: {loss_eta.item()}, CL: {loss_cl.item()}"
                    else:
                        desc = f"L1: {loss_eta.item()}"
                    tqdm_loader.set_description(
                        f'{phase} epoch: {epoch}, {phase} loss: {(running_loss[phase] / steps) :.8f}, '
                        + desc
                    )
                    with torch.no_grad():
                
                        # 3. Exponentiate safely
                        predictions.append(output.detach().cpu())
                        targets.append(truth_data.detach().cpu())

                    running_loss[phase] += loss.item() * truth_data.size(0)

                torch.cuda.empty_cache()
                gc.collect()
                    
                predictions = torch.cat(predictions).numpy()
                targets = torch.cat(targets).numpy()
                scores = calculate_metrics(predictions.reshape(predictions.shape[0], -1),
                                           targets.reshape(targets.shape[0], -1), args, plot=epoch % 5 == 0, **kwargs)
                
                with open(model_folder+"/output.txt", "a") as f:
                    f.write(f'{phase} epoch: {epoch}, {phase} loss: {running_loss[phase] / steps}\n')
                    f.write(str(scores))
                    f.write('\n')
                    f.write(str(time.time()))
                    f.write("\n\n")
                print(scores)
                msg.append(f"{phase} epoch: {epoch}, {phase} loss: {running_loss[phase] / steps}\n {scores}\n")
                
                if phase == 'val':
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Current LR: {current_lr:.6e}")
                    if scores['MAE'] < best_mae:
                        best_mae = scores['MAE']
                        save_dict.update(
                            state_dict=copy.deepcopy(model.state_dict()),
                            epoch=epoch,
                            optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
                            scheduler_state_dict=copy.deepcopy(scheduler.state_dict())
                        )
                        save_model(f"{model_folder}/best_model.pkl", **save_dict)
                        print(f"New best MAE {best_mae} at epoch {epoch}, model saved.")
                    else:
                        print(f"Current MAE {scores['MAE']} more than best MAE {best_mae}")

            scheduler.step(running_loss['val'])
    finally:
        time_elapsed = time.perf_counter() - since
        hours, remainder = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Print with seconds rounded to 2 decimal places
        print(f"cost {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        save_model(f"{model_folder}/best_model.pkl", **save_dict)
        save_model(f"{model_folder}/final_model.pkl",
                   **{'state_dict': copy.deepcopy(model.state_dict()),
                      'epoch': epoch,
                      'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                      'scheduler_state_dict': copy.deepcopy(scheduler.state_dict())
                      })