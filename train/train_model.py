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

from utils.metric import calculate_metrics
from utils.util import save_model, to_var, get_warmup_cosine_scheduler, LossBalancer


def train_model(model:         nn.Module,
                data_loaders:  Dict[str, DataLoader],
                loss_func:     callable,
                optimizer,
                model_folder:  str,
                args,
                start_epoch:   int  = 0,
                global_step:   int  = 0,
                best_mae:      float = 1e9,
                total_steps:   int  = None,
                warmup_steps:  int  = None,
                scheduler             = None,
                loss_balancer: LossBalancer = None,
                **kwargs):

    phases = ['train', 'val']
    since  = time.perf_counter()

    # CUDA AMP only
    use_amp = str(args.device).startswith("cuda")

    scaler = torch.amp.GradScaler(enabled=use_amp)

    if loss_balancer is None:
        loss_balancer = LossBalancer()

    save_dict = {
        'state_dict': copy.deepcopy(model.state_dict()),
        'epoch':       start_epoch,
        'global_step': global_step,
        'best_mae':    best_mae,
        'total_steps': total_steps,
        'warmup_steps':warmup_steps,
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

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps = 0
                predictions = []
                targets = []

                tqdm_loader = tqdm(data_loaders[phase], mininterval=3)

                for features, truth_data in tqdm_loader:

                    steps += truth_data.size(0)

                    features   = to_var(features, args.device)
                    truth_data = to_var(truth_data, args.device)

                    truth_data = torch.clamp(truth_data, min=0.0)

                    with torch.set_grad_enabled(phase == 'train'):

                        with torch.amp.autocast(
                            device_type="cuda",
                            enabled=use_amp
                        ):

                            output, logits, soft_weights = model(
                                features,
                                truth_data
                            )

                            loss_eta = loss_func(
                                truth=truth_data,
                                predict=output
                            )

                            loss = loss_eta

                        if phase == 'train':

                            optimizer.zero_grad()

                            if use_amp:

                                scaler.scale(loss).backward()

                                scaler.unscale_(optimizer)

                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    1.0
                                )

                                scaler.step(optimizer)

                                old_scaler = scaler.get_scale()

                                scaler.update()

                                new_scaler = scaler.get_scale()

                                if (
                                    scheduler is not None and
                                    new_scaler >= old_scaler
                                ):
                                    scheduler.step()

                            else:

                                loss.backward()

                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    1.0
                                )

                                optimizer.step()

                                if scheduler is not None:
                                    scheduler.step()

                            global_step += 1

                    desc = f"L1: {loss_eta.item():.4f}"

                    tqdm_loader.set_description(
                        f"{phase} epoch {epoch} | "
                        f"loss: {running_loss[phase] / max(steps, 1):.6f} | "
                        f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
                        f"{desc}"
                    )

                    with torch.no_grad():
                        predictions.append(output.detach().cpu())
                        targets.append(truth_data.detach().cpu())

                    running_loss[phase] += (
                        loss.item() * truth_data.size(0)
                    )

                gc.collect()

                predictions = torch.cat(predictions).numpy()
                targets     = torch.cat(targets).numpy()

                scores = calculate_metrics(
                    predictions.reshape(predictions.shape[0], -1),
                    targets.reshape(targets.shape[0], -1),
                    args,
                    plot=(epoch % 5 == 0),
                    **kwargs
                )

                epoch_loss = running_loss[phase] / steps

                log_line = (
                    f"{phase} epoch: {epoch} | "
                    f"loss: {epoch_loss:.6f} | "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e}\n"
                    f"{scores}\n{time.time()}\n\n"
                )

                with open(
                    os.path.join(model_folder, "output.txt"),
                    "a"
                ) as f:
                    f.write(log_line)

                print(scores)

                if phase == 'val':

                    print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

                    if scores['MAE'] < best_mae:

                        best_mae = float(scores['MAE'])

                        save_dict.update(
                            state_dict           = copy.deepcopy(model.state_dict()),
                            epoch                = epoch,
                            global_step          = global_step,
                            best_mae             = best_mae,
                            total_steps          = total_steps,
                            warmup_steps         = warmup_steps,
                            optimizer_state_dict = copy.deepcopy(optimizer.state_dict()),
                        )

                        if scheduler is not None:
                            save_dict['scheduler_state_dict'] = copy.deepcopy(
                                scheduler.state_dict()
                            )

                        save_model(
                            os.path.join(model_folder, "best_model.pkl"),
                            **save_dict
                        )

                        print(
                            f"New best MAE {best_mae:.4f} at epoch {epoch}"
                        )

                    else:
                        print(
                            f"MAE {scores['MAE']:.4f} "
                            f"(best {best_mae:.4f})"
                        )

            save_payload = dict(
                state_dict           = copy.deepcopy(model.state_dict()),
                epoch                = epoch + 1,
                global_step          = global_step,
                best_mae             = float(best_mae),
                total_steps          = total_steps,
                warmup_steps         = warmup_steps,
                optimizer_state_dict = copy.deepcopy(optimizer.state_dict()),
            )

            if scheduler is not None:
                save_payload['scheduler_state_dict'] = copy.deepcopy(
                    scheduler.state_dict()
                )

            save_model(
                os.path.join(model_folder, "final_model.pkl"),
                **save_payload
            )

    finally:

        elapsed = time.perf_counter() - since

        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)

        print(
            f"Training complete: "
            f"{int(h)}h {int(m)}m {s:.2f}s"
        )

        save_model(
            os.path.join(model_folder, "best_model.pkl"),
            **save_dict
        )