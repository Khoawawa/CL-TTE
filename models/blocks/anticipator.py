import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.LayerNormGRU import LayerNormGRU

class GRUAnticipator(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.gru = LayerNormGRU(input_dim=d_model, hidden_dim=d_model, num_layers=num_layers)
        self.init_proj = nn.Linear(d_model, num_layers * d_model)
        self.fuse = nn.Linear(2 * d_model, d_model)
    def masked_mean_pooling(self, h, mask):
        masked_h = h * mask.unsqueeze(-1).float()  # (B, T, D)
        sum_h = masked_h.sum(dim=1)  # (B, D)
        count = mask.sum(dim=1).unsqueeze(-1)  # (B, 1)
        mean_h = sum_h / count.clamp(min=1)  # (B, D)
        return mean_h
    def forward(self, x, z, lens):
        # x: (B, T, D)
        # z: (B, D) - context vector
        # lens: (B,) - lengths of each sequence
        B, T, D = x.shape 
        
        gru_input = x.transpose(0, 1).contiguous()  # (T, B, D)
        h0 = self.init_proj(z).view(self.num_layers, B, D)  # (num_layers, B, D)
        y, hy = self.gru(gru_input, lens, hx=h0)
        
        summary = self.masked_mean_pooling(y.transpose(0, 1), mask=torch.arange(T, device=lens.device).unsqueeze(0) < lens.unsqueeze(1))  # (B, D)
        
        final_driver_state = self.fuse(torch.cat([summary, z], dim=-1))  # (B, D)
        return final_driver_state