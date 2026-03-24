import torch
import torch.nn as nn
from models.blocks.encoder import SegmentEncoder, ContrastiveEncoder

class Cl_TTE(nn.Module):
    def __init__(self, d_model, nhead,nlayer,moco_queue_size=1024, moco_temperature=0.05):
        super().__init__()
        self.segment_encoder = SegmentEncoder(
            d_model= d_model
        )
        
        self.contrastive = ContrastiveEncoder(
            d_model=d_model,
            nhead=nhead,
            nlayer=nlayer,
            queue_size=moco_queue_size,
            temperature=moco_temperature
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
        links = inputs['links']
        dateinfo = inputs['dateinfo']
        lens = inputs['lens']
        
        segment_rep, datetimerep = self.segment_encoder(links,dateinfo,lens)  # (B, T, D)
        
        h, l_cl = self.contrastive(segment_rep, lens,args.mask_prob, args.noise, args.r, y_true) # (B, T, D)
        
        h_time = torch.concat([h, datetimerep], dim=-1)
        
        segment_t = self.regression_mlp(h) # (B, T, 1)
        
        max_len = segment_t.shape[1]
        mask = torch.arange(max_len, device=lens.device).unsqueeze(0) < lens.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        
        segment_t = segment_t * mask

        t = segment_t.sum(dim=1) # (B, 1)
        
        return t, l_cl