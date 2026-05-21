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
from utils.util import save_model, to_var, get_warmup_cosine_scheduler_with_floor, LossBalancer
from torch.cuda import memory_allocated, memory_reserved, reset_peak_memory_stats
from utils.prepare import create_loss

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag
def profile_components(model, data_loader, device):
    model.train()
    batch, truth = next(iter(data_loader))
    features   = to_var(batch, device)
    truth_data = truth.to(device)

    x         = features['links_clean']
    x_aug     = features['links_aug']
    dateinfo  = features['dateinfo']
    culm_len  = features['culm_len']
    mask      = features.get('src_key_padding_mask', None)

    def measure(label, fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        out = fn()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated(device)
        peak  = torch.cuda.max_memory_allocated(device)
        print(f"{label:35s} alloc delta: {(after-before)/1e9:.3f}GB  peak: {(peak-before)/1e9:.3f}GB")
        return out

    with torch.amp.autocast(device_type='cuda'):
        dtrep     = measure("time_encoder",     lambda: model.contrastive_module.time_encoder(dateinfo))
        orig_repr = measure("segment_encoder(x)",    lambda: model.contrastive_module.segment_encoder(x))
        aug_repr  = measure("segment_encoder(x_aug)",lambda: model.contrastive_module.segment_encoder(x_aug))
        orig_film = measure("film(orig)",        lambda: model.contrastive_module.film(orig_repr, dtrep))
        aug_film  = measure("film(aug)",         lambda: model.contrastive_module.film(aug_repr,  dtrep))

        h, trip_orig = measure("encode(orig)",   lambda: model.contrastive_module.contrastive_encoder.encode(orig_film, mask))
        
        with torch.no_grad():
            _, trip_aug = measure("encode(aug) no_grad", lambda: model.contrastive_module.contrastive_encoder.encode(aug_film, mask))
        
        measure("contrastive_loss",  lambda: model.contrastive_module.contrastive_encoder.calculate_contrastive_loss(trip_orig, trip_aug))
        measure("after_proj",        lambda: model.contrastive_module.after_proj(torch.cat([h, culm_len], dim=-1)))

def profile_single_batch(model, data_loader, device, args):
    model.train()
    batch, truth = next(iter(data_loader))
    
    # move to GPU
    features   = to_var(batch, device)
    truth_data = truth.to(device)
    
    reset_peak_memory_stats(device)
    baseline = memory_allocated(device)
    print(f"baseline (model weights):     {baseline/1e9:.3f} GB")

    # --- forward only ---
    reset_peak_memory_stats(device)
    with torch.amp.autocast(device_type='cuda'):
        output, loss_cl, cl_metric = model(features, truth_data)
    
    after_fwd = memory_allocated(device)
    peak_fwd  = torch.cuda.max_memory_allocated(device)
    print(f"after forward:                {after_fwd/1e9:.3f} GB")
    print(f"peak during forward:          {peak_fwd/1e9:.3f} GB")
    print(f"forward activations:          {(peak_fwd - baseline)/1e9:.3f} GB")

    # --- backward ---
    reset_peak_memory_stats(device)
    loss_func = create_loss(args)
    loss_func(truth=truth_data, predict=output).backward()
    
    peak_bwd = torch.cuda.max_memory_allocated(device)
    print(f"peak during backward:         {peak_bwd/1e9:.3f} GB")
    print(f"backward overhead:            {(peak_bwd - after_fwd)/1e9:.3f} GB")

    # --- what's left after del ---
    del output, loss_cl
    torch.cuda.empty_cache()
    after_del = memory_allocated(device)
    print(f"after del + empty_cache:      {after_del/1e9:.3f} GB")
    print(f"leaked this batch:            {(after_del - baseline)/1e9:.3f} GB")

# profile_single_batch(model, data_loaders['train'], args.device)
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
        total_steps     = steps_per_epoch * getattr(args, 'max_epochs', 50)
        warmup_steps    = max(1, int(0.05 * total_steps))
        scheduler       = get_warmup_cosine_scheduler_with_floor(optimizer, warmup_steps, total_steps)
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
    model.use_contrastive = True
    
    profile_single_batch(model, data_loaders['train'], args.device, args)
    profile_components(model, data_loaders['train'], args.device)
    try:
        for epoch in range(start_epoch, args.epochs):
            running_loss = {phase: 0.0 for phase in phases}
            
            for phase in phases:
                args.phase = phase
                model.train() if phase == 'train' else model.eval()
                
                n_samples   = len(data_loaders[phase].dataset)
                predictions = np.empty(n_samples, dtype=np.float32)
                targets_arr = np.empty(n_samples, dtype=np.float32)
                cursor      = 0
                steps = 0
                tqdm_loader = tqdm(data_loaders[phase], mininterval=3)

                for features, truth_data in tqdm_loader:
                    steps     += truth_data.size(0)
                    features    = to_var(features,   args.device)
                    truth_data  = to_var(truth_data,  args.device)

                    # Dynamic Warmup Shift for Contrastive Task
                    # if (phase == 'train' and not model.use_contrastive and (global_step >= warmup_steps // 2)):
                    #     model.use_contrastive = True
                    #     loss_balancer.reset()
                    #     print(f"\n Step {global_step}: Warmup completed. Enabling contrastive learning.\n")
                        
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.amp.autocast(device_type='cuda' if 'cuda' in str(args.device) else 'cpu', enabled=True):
                            output, loss_cl, cl_metric = model(features, truth_data)
                            loss_eta        = loss_func(truth=truth_data, predict=output)

                            # Explicit variable tracking to clean out validation leaks
                            if phase == 'train' and loss_cl is not None:
                                loss = loss_eta + args.beta * loss_cl
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
                    if cl_val is not None and cl_metric is not None:
                        scale = loss_balancer.get_scale()
                        desc = f"L1: {loss_eta.item():.4f}  CL: {cl_val:.4f}"
                        for k,v in cl_metric.items():
                            if isinstance(v, float):
                                desc += f" {k}: {v:.4f}"
                            else:
                                desc += f" {k}: {v}"
                        desc += f" scale: {scale:.4f}"
                    else:
                        desc = f"L1: {loss_eta.item():.4f}"

                    tqdm_loader.set_description(
                        f"{phase} epoch {epoch} | "
                        f"loss: {running_loss[phase] / max(steps, 1):.6f} | "
                        f"lr: {optimizer.param_groups[0]['lr']:.2e} | cl_lr: {optimizer.param_groups[1]['lr']:.2e} | {desc}"
                    )

                    # Optimized performance tracking: Append tensors on GPU directly
                    with torch.no_grad():
                        pred_np = output.detach().float().cpu().numpy().reshape(-1)
                        tgt_np  = truth_data.detach().float().cpu().numpy().reshape(-1)
                        B_actual = len(pred_np)
                        predictions[cursor:cursor + B_actual] = pred_np
                        targets_arr[cursor:cursor + B_actual] = tgt_np
                        cursor += B_actual


                    running_loss[phase] += loss.item() * truth_data.size(0)
                    
                    del output, loss, loss_eta, loss_cl
                # Clean execution states before pushing to CPU metric suites
                torch.cuda.empty_cache()
                gc.collect()
                
                scores      = calculate_metrics(
                    predictions[:cursor].reshape(-1, 1),
                    targets_arr[:cursor].reshape(-1, 1),
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