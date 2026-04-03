import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.LayerNormGRU import LayerNormGRU

class GRUAnticipator(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.gru = LayerNormGRU(input_size=d_model * 2, hidden_size=d_model, num_layers=num_layers)
    
    def forward(self, x, z, lens):
        # x: (B, T, D)
        # z: (B, D) - context vector
        # lens: (B,) - lengths of each sequence
        B, T, D = x.shape 
        z_expanded = z.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        gru_input = torch.cat([x, z_expanded], dim=-1)  # (B, T, 2D)
        
        gru_input = gru_input.transpose(0, 1).contiguous()  # (T, B, 2D)
        output = self.gru(gru_input, lens)  # (T, B, D)
         = output.transpose(0, 1).contiguous()  # (B, T, D)
        
        return output