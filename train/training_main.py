import os
import sys
import shutil
from tqdm import tqdm

import torch
from torch import optim
import numpy as np

from train.train_model import train_model
from utils.prepare import create_model, create_loss
from utils.prepare import load_datadict, load_datadoct_pre
from utils.prepare import load_test_datadict
from utils.metric import calculate_metrics
from utils.util import to_var, LossBalancer, get_warmup_cosine_scheduler_with_floor, save_model, load_checkpoint
import time

def test_model(model, data_loader, args):
    model.eval()
    predictions = list()
    targets = list()
    inds = list()
    tqdm_loader = tqdm(data_loader)
    for step, (features, truth_data) in enumerate(tqdm_loader):
        if isinstance(features, dict) and 'inds' in features.keys():
            inds.append(features['inds'])
        features = to_var(features, args.device)
        truth_data = to_var(truth_data, args.device)

        outputs, _ = model(features, args)

        targets.append(truth_data.cpu().numpy())
        predictions.append(outputs.cpu().detach().numpy())
    pre2 = np.concatenate(predictions).squeeze()
    tar2 = np.concatenate(targets)
    if len(inds) > 0:
        print(f"test size: {len(inds)}")
        print(f"test traj ids of a batch: {inds[0]}")
        inds = np.concatenate(inds)
    else:
        inds = None
    metric = calculate_metrics(pre2, tar2, args, plot=True, inds=inds)
    print(metric)
    with open(f'{args.absPath}/data/result_{args.model}.txt', 'a') as f:
        f.write(time.strftime("%m/%d %H:%M:%S",time.localtime(time.time())))
        f.write(f"epoch:{args.epochs} lr:{args.lr}\ndataset:{args.dataset} identify:{args.identify}\nloss:{args.loss}\n")
        f.write(f"{args.model_config}\n")
        f.write(f"{args.data_config}\n")
        f.write(f"{metric}\n\n")

    np.save(os.path.join(args.model_folder, "result.npy"), np.asarray([pre2, inds]))

def train_main(args):
    if args.model == 'None':
        print('No chosen model')
        sys.exit(0)
 
    print(f"{args.mode} {args.model}_{args.identify} on {args.dataset}")
 
    load_datadoct_pre(args)
    data_loaders, scaler = load_datadict(args)
    args.scaler = scaler
 
    model      = create_model(args).to(args.device)
    loss_func  = create_loss(args)
    model_folder = (
        f"{args.absPath}/data/save_models/"
        f"{args.model}_{args.identify}_{args.dataset}"
    )
    args.model_folder = model_folder
 
    print(f"loss:         {args.loss}")
    print(f"model config: {args.model_config}")
    print(f"data config:  {args.data_config}")
 
    def make_optimizer():
        contrastive_params = list(model.contrast_enc.parameters())
        contrastive_ids = set(id(p) for p in contrastive_params)
        other_params = [p for p in model.parameters() if id(p) not in contrastive_ids]
        
        if args.optim == "Adam":
            return optim.Adam([
                {'params': other_params,       'lr': args.lr},          # 1e-3
                {'params': contrastive_params, 'lr': args.lr},    # 5e-4
            ])
        elif args.optim == "AdamW":
            return optim.AdamW([
                {'params': other_params,       'lr': args.lr,       'weight_decay': args.weight_decay, 'betas': (0.9, 0.95)},
                {'params': contrastive_params, 'lr': args.lr, 'weight_decay': args.weight_decay, 'betas': (0.9, 0.95)},
            ])
        raise NotImplementedError(f"Unknown optimizer: {args.optim}")
 
    # ------------------------------------------------------------------ train
    if args.mode == 'train':
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder, ignore_errors=True)
        os.makedirs(model_folder, exist_ok=True)
 
        optimizer = make_optimizer()
        train_model(
            model=model, data_loaders=data_loaders,
            loss_func=loss_func, optimizer=optimizer,
            model_folder=model_folder, args=args,
        )
 
    # ----------------------------------------------------------------- resume
    elif args.mode == 'resume':
        ckpt_path = os.path.join(model_folder, 'final_model.pkl')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
 
        optimizer     = make_optimizer()
        loss_balancer = LossBalancer()
 
        # build a temporary scheduler so load_checkpoint can restore its state
        # total_steps will be corrected from the checkpoint
        tmp_scheduler = get_warmup_cosine_scheduler_with_floor(optimizer, 1, 1)
 
        start_epoch, global_step, best_mae, total_steps, warmup_steps = load_checkpoint(
            ckpt_path, model, optimizer, tmp_scheduler, loss_balancer, args.device
        )
 
        # rebuild scheduler with the saved total_steps so the LR curve
        # is identical to the original training run
        if total_steps is not None and warmup_steps is not None:
            scheduler = get_warmup_cosine_scheduler_with_floor(optimizer, warmup_steps, total_steps)
            scheduler.load_state_dict(tmp_scheduler.state_dict())
        else:
            # fallback: recompute from current dataset size and remaining epochs
            steps_per_epoch = len(data_loaders['train'])
            total_steps     = steps_per_epoch * args.epochs
            warmup_steps    = max(1, int(0.05 * total_steps))
            scheduler       = get_warmup_cosine_scheduler_with_floor(optimizer, warmup_steps, total_steps)
            print("Warning: total_steps not found in checkpoint, recomputed from args.")
 
        print(f"Resumed from epoch {start_epoch}, step {global_step}, best MAE {best_mae:.4f}")
 
        train_model(
            model=model, data_loaders=data_loaders,
            loss_func=loss_func, optimizer=optimizer,
            model_folder=model_folder, args=args,
            start_epoch=start_epoch,
            global_step=global_step,
            best_mae=best_mae,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            scheduler=scheduler,
            loss_balancer=loss_balancer,
        )