import math

import torch
import torch.nn as nn
from models.base.PositionalEncoding import CyclicalTimeEncoding, PositionalEncoding1D
from models.blocks.cl import MSM, ReCo
from models.blocks.poi import PoiResGatedFilMEncoder
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
    