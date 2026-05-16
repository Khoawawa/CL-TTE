import math
import os
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

class StandardScaler2:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)

def to_var(var, device=0):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.to(device)
        return var
    if isinstance(var,np.ndarray):
        var_tensor = torch.from_numpy(var)
        return to_var(var_tensor,device)
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key], device)
        return var
    if isinstance(var, list):
        var = list(map(lambda x: to_var(x, device), var))
        return var
    
class LossBalancer:
    """
    EMA-smoothed dynamic loss balancer.
    Keeps l_cl at the same magnitude as l_eta by computing a smoothed
    scale = ema(l_eta) / ema(l_cl) and clamping it to a safe range.
    """
    def __init__(self, ema_decay=0.9, clamp=(0.01, 50.0)):
        self.ema_decay = ema_decay
        self.clamp     = clamp
        self.ema_eta   = None
        self.ema_cl    = None
 
    def state_dict(self):
        return {
            'ema_eta': self.ema_eta,
            'ema_cl':  self.ema_cl,
        }
    def reset(self):
        self.ema_eta = None
        self.ema_cl  = None
    def load_state_dict(self, d):
        self.ema_eta = d.get('ema_eta', None)
        self.ema_cl  = d.get('ema_cl',  None)
 
    def __call__(self, loss_eta: torch.Tensor,
                 loss_cl:  torch.Tensor,
                 beta:     float) -> torch.Tensor:
 
        with torch.no_grad():
            val_eta = loss_eta.item()
            val_cl  = loss_cl.item()  if loss_cl is not None else 0.0
 
            if self.ema_eta is None:
                self.ema_eta = val_eta
                self.ema_cl  = val_cl
            else:
                d = self.ema_decay
                self.ema_eta = d * self.ema_eta + (1 - d) * val_eta
                self.ema_cl  = d * self.ema_cl  + (1 - d) * val_cl
 
            scale = self.ema_eta / (self.ema_cl + 1e-6)
            scale = max(self.clamp[0], min(self.clamp[1], scale))
 
        if loss_cl is None:
            return loss_eta
 
        return beta * loss_eta + (1 - beta) * loss_cl * scale
 
 
# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------
 
def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup for `warmup_steps` steps, then cosine decay to 0.
    Call scheduler.step() once per optimizer step (not per epoch).
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
 
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_model(path: str, **kwargs):
    torch.save(kwargs, path)
 
 
def load_checkpoint(path: str, model: nn.Module, optimizer, scheduler,
                    loss_balancer: LossBalancer, device):
    
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    
    ckpt = torch.load(path, map_location=device, weights_only=False)
 
    model.load_state_dict(ckpt['state_dict'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
 
    if 'scheduler_state_dict' in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
 
    if 'balancer_state_dict' in ckpt:
        loss_balancer.load_state_dict(ckpt['balancer_state_dict'])
 
    start_epoch  = ckpt.get('epoch',        0)
    global_step  = ckpt.get('global_step',  0)
    best_mae     = ckpt.get('best_mae',     1e9)
    total_steps  = ckpt.get('total_steps',  None)
    warmup_steps = ckpt.get('warmup_steps', None)
 
    return start_epoch, global_step, best_mae, total_steps, warmup_steps