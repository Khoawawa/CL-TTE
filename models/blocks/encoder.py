import math

import torch
import torch.nn as nn
from models.base.PositionalEncoding import CyclicalTimeEncoding, PositionalEncoding1D
from models.blocks.cl import MSM, ReCo
from models.blocks.poi import PoiResGatedFilMEncoder

class ContrastiveEncoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, nlayer=4):
        super().__init__()
        self.msm = MSM(d_model,nhead,dropout,nlayer)
    def create_pos_mask(self, x, y_true, r_percentile=0.2):
        if y_true.dim() == 2:
            y_true = y_true.squeeze(-1)
        y_true = y_true.detach()
        B = y_true.size(0)

        dist_orig = torch.abs(y_true.unsqueeze(0) - y_true.unsqueeze(1))
        mask = ~torch.eye(B, dtype=torch.bool, device=y_true.device)
        dist_flat = dist_orig[mask]
        
        r = torch.quantile(dist_flat, r_percentile)
        r = torch.clamp(r, min=1e-3)

        y_all = torch.cat([y_true, y_true], dim=0) 
        dist = torch.abs(y_all.unsqueeze(0) - y_all.unsqueeze(1))
        
        pos_mask = (dist <= r).float()
        pos_mask.fill_diagonal_(0)

        num_pos = pos_mask.sum(dim=1)
        if (num_pos == 0).any():
            dist_no_self = dist + torch.eye(dist.size(0), device=dist.device) * 1e4
            nn_idx = dist_no_self.argmin(dim=1)
            pos_mask[torch.arange(dist.size(0)), nn_idx] = 1.0

        return pos_mask
    
    def forward(self, x, src_key_padding_mask=None):
        return self.msm(x, src_key_padding_mask)
class SegmentEncoder(nn.Module):
    def __init__(self,n_poi_groups, d_model=128):
        
        super().__init__()

        self.highwayembed = nn.Embedding(17, 6, padding_idx=0)
        self.gpsembed = nn.Linear(4,16)
        
        self.weekembed = nn.Embedding(8, 3)
        self.dateembed = PositionalEncoding1D(10)
        self.timeembed = PositionalEncoding1D(d_model=20)
        
        self.poi_embed = PoiResGatedFilMEncoder(n_poi_groups=n_poi_groups, embed_dim=32, n_layers=2)
        mlp_in_dim = 2 + 12 + 16 + 32 # = 2 + 6 + 16 + 32 // 8 = 
        
        self.datetime_dim = 3 + 10 + 20
        
        self.represent = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_in_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim * 2, d_model)
        )
        
    def forward(self, links, dateinfo): 
        # links: (B, T, 17) [HighwayID1, HighwayID2, Len, CumLen, Lat1, Lon1, Lat2, Lon2, poi*9]
        # dateinfo: (B, 3)
        # poi_counts: (B, T, n_poi_groups)
        # lens: (B,)
        B, T, _ = links.shape
        
        weekrep   = self.weekembed(dateinfo[:, 0].long())
        daterep   = self.dateembed(dateinfo[:, 1])
        timerep   = self.timeembed(dateinfo[:, 2])
        
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1)
        datetimerep_expand = datetimerep.unsqueeze(1).expand(-1,T, -1) # (B,T,seq_hidden_dim)
        # spatial features
        highwayrep1 = self.highwayembed(links[:, :, 0].long()) # 6
        highwayrep2 = self.highwayembed(links[:, :, 1].long()) # 6
        highwayrep = torch.cat([highwayrep1, highwayrep2], dim=-1)
        
        gpsrep = torch.tanh(self.gpsembed(links[:, :, 4:8].float())) # 16
        
        # poi features
        poirep = self.poi_embed(links[:, :, 8:].float(), datetimerep_expand)
        features = torch.cat([links[..., 2:4], gpsrep,highwayrep, poirep], dim=-1) # 2 + 5 + 16 + 33 
        
        features_proj = self.represent(features) # (B,T,seq_hidden_dim)
        
        return features_proj, datetimerep
    