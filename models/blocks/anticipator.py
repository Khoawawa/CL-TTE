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
    
    def forward(self, x, z, lens):
        # x: (B, T, D)
        # z: (B, D) - context vector
        # lens: (B,) - lengths of each sequence
        B, T, D = x.shape 
        
        gru_input = x.transpose(0, 1).contiguous()  # (T, B, 2D)
        h0 = z.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()  # (num_layers, B, D)
        y, hy = self.gru(gru_input, lens, hx=h0)
        
        final_driver_state = hy[-1]  # (B, D)
        
        return final_driver_state