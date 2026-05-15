import copy
import time
from typing import Dict
import gc
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.main_model import Cl_TTE
from utils.metric import calculate_metrics
from utils.util import save_model, to_var, get_warmup_cosine_scheduler, LossBalancer


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag
        

def train_model(model:          Cl_TTE,
                data_loaders:   Dict[str, DataLoader],
                loss_func:      callable,
                optimizer,
                model_folder:   str,
                args,
                start_epoch:    int  = 0,
                global_step:    int  = 0,
                best_mae:       float = 1e9,
                total_steps:    int  = None,
                warmup_steps:   int  = None,
                scheduler             = None,
                loss_balancer: LossBalancer = None,
                **kwargs):

    # ---- Build scheduler if not provided (fresh training) -----------------
    if scheduler is None:
        steps_per_epoch = len(data_loaders['train'])
        total_steps     = steps_per_epoch * getattr(args, 'max_epochs', 20)
        warmup_steps    = max(1, int(0.1 * total_steps))
        scheduler       = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
        print(f"Scheduler: warmup {warmup_steps} steps, total {total_steps} steps")

    if loss_balancer is None:
        loss_balancer = LossBalancer()

    phases = ['train', 'val']
    since  = time.perf_counter()
    scaler = torch.amp.GradScaler()

    save_dict = {
        'state_dict': copy.deepcopy(model.state_dict()),
        'epoch':       start_epoch,
        'global_step': global_step,
        'best_mae':    best_mae,
        'total_steps': total_steps,
        'warmup_steps': warmup_steps,
    }

    with open(os.path.join(model_folder, "output.txt"), "a") as f:
        f.write(str(model))
        f.write("\n\n")

    print(f"Starting LR: {optimizer.param_groups[0]['lr']:.2e}")

    try:
        for epoch in range(start_epoch, args.epochs):
            running_loss = {phase: 0.0 for phase in phases}

            for phase in phases:
                args.phase = phase
                model.train() if phase == 'train' else model.eval()

                steps, predictions, targets = 0, [], []
                tqdm_loader = tqdm(data_loaders[phase], mininterval=3)

                for features, truth_data in tqdm_loader:
                    steps     += truth_data.size(0)
                    features    = to_var(features,   args.device)
                    truth_data  = to_var(truth_data,  args.device)

                    # Dynamic Warmup Shift for Contrastive Task
                    if (phase == 'train' and not model.use_contrastive and (global_step >= warmup_steps // 2)):
                        model.use_contrastive = True
                        loss_balancer.reset()
                        print(f"\n Step {global_step}: Warmup completed. Enabling contrastive learning.\n")
                        
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.amp.autocast(device_type='cuda' if 'cuda' in str(args.device) else 'cpu', enabled=True):
                            output, loss_cl = model(features, truth_data)
                            loss_eta        = loss_func(truth=truth_data, predict=output)

                            # Explicit variable tracking to clean out validation leaks
                            if phase == 'train' and loss_cl is not None:
                                loss = loss_balancer(loss_eta, loss_cl, args.beta)
                            else:
                                loss = loss_eta
                                loss_cl = None

                        if phase == 'train':
                            optimizer.zero_grad()
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            
                            # Optimized Scaler + Scheduler block execution
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            global_step += 1

                    # TQDM live logging updates
                    cl_val = loss_cl.item() if loss_cl is not None else None
                    if cl_val is not None:
                        desc = f"L1: {loss_eta.item():.4f}  CL: {cl_val:.4f}"
                    else:
                        desc = f"L1: {loss_eta.item():.4f}"

                    tqdm_loader.set_description(
                        f"{phase} epoch {epoch} | "
                        f"loss: {running_loss[phase] / max(steps, 1):.6f} | "
                        f"lr: {optimizer.param_groups[0]['lr']:.2e} | {desc}"
                    )

                    # Optimized performance tracking: Append tensors on GPU directly
                    with torch.no_grad():
                        predictions.append(output.detach().cpu())
                        targets.append(truth_data.detach().cpu())

                    running_loss[phase] += loss.item() * truth_data.size(0)

                # Clean execution states before pushing to CPU metric suites
                torch.cuda.empty_cache()
                gc.collect()

                # Concatenate on GPU exactly once, then drop to CPU arrays
                predictions = torch.cat(predictions, dim=0).numpy()
                targets     = torch.cat(targets, dim=0).numpy()
                
                scores      = calculate_metrics(
                    predictions.reshape(predictions.shape[0], -1),
                    targets.reshape(targets.shape[0], -1),
                    args,
                    plot=(epoch % 5 == 0),
                    **kwargs
                )

                epoch_loss = running_loss[phase] / steps
                log_line   = (
                    f"{phase} epoch: {epoch} | loss: {epoch_loss:.6f} | "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e}\n"
                    f"{scores}\n{time.time()}\n\n"
                )
                with open(os.path.join(model_folder, "output.txt"), "a") as f:
                    f.write(log_line)
                print(scores)

                if phase == 'val':
                    print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

                    if scores['MAE'] < best_mae:
                        best_mae = float(scores['MAE'])
                        save_dict.update(
                            state_dict            = copy.deepcopy(model.state_dict()),
                            epoch                 = epoch,
                            global_step           = global_step,
                            best_mae              = best_mae,
                            total_steps           = total_steps,
                            warmup_steps          = warmup_steps,
                            optimizer_state_dict  = copy.deepcopy(optimizer.state_dict()),
                            scheduler_state_dict  = copy.deepcopy(scheduler.state_dict()),
                            balancer_state_dict   = copy.deepcopy(loss_balancer.state_dict()),
                        )
                        save_model(os.path.join(model_folder, "best_model.pkl"), **save_dict)
                        print(f"New best MAE {best_mae:.4f} at epoch {epoch}")
                    else:
                        print(f"MAE {scores['MAE']:.4f} (best {best_mae:.4f})")

            # Save tracking states strictly for resumption purposes
            save_model(
                os.path.join(model_folder, "final_model.pkl"),
                state_dict           = copy.deepcopy(model.state_dict()),
                epoch                = epoch + 1,
                global_step          = global_step,
                best_mae             = float(best_mae),
                total_steps          = total_steps,
                warmup_steps         = warmup_steps,
                optimizer_state_dict = copy.deepcopy(optimizer.state_dict()),
                scheduler_state_dict = copy.deepcopy(scheduler.state_dict()),
                balancer_state_dict  = copy.deepcopy(loss_balancer.state_dict()),
            )

    finally:
        elapsed = time.perf_counter() - since
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        print(f"Training session ended: {int(h)}h {int(m)}m {s:.2f}s")
        # Kept as runtime safety backup so Ctrl+C breaks won't corrupt your actual validation best file
        save_model(os.path.join(model_folder, "emergency_backup_state.pkl"), **save_dict)