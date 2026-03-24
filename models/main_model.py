import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.encoder import SegmentEncoder, ContrastiveEncoder

class Cl_TTE(nn.Module):
    def __init__(self, d_model, nhead, nlayer):
        super().__init__()
        
        self.segment_encoder = SegmentEncoder(
            d_model= d_model
        )
        
        self.contrastive = ContrastiveEncoder(
            d_model=d_model,
            nhead=nhead,
            nlayer=nlayer
        )
        
        mlp_in_dim = d_model + self.segment_encoder.datetime_dim
        self.regression_mlp = nn.Sequential(
            nn.LayerNorm(mlp_in_dim),
            nn.Linear(mlp_in_dim, mlp_in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim//2, 1)
        )
        
    def forward(self, inputs, y_true, args):
        # inputs: 
        # links: [B, T, 7] -> (highway, len, culm_len, start_lat, start_lon, end_lat, end_lon)
        # dateinfo : [B, 3]
        # valid_mask: [B, T] 
        # lens: [B]
        # y_true: [B, 1] --> logged gt
        links = inputs['links']
        dateinfo = inputs['dateinfo']
        lens = inputs['lens']
        
        segment_rep, datetimerep = self.segment_encoder(links,dateinfo,lens)  # (B, T, D)
        
        z, l_cl = self.contrastive(segment_rep, lens,args.mask_prob, args.noise, args.r, y_true) # (B, T, D)
        
        z_time = torch.concat([z, datetimerep], dim=-1) # (B,D + 33)
        
        t = self.regression_mlp(z_time) # (B, 1)

        
        return t, l_cl