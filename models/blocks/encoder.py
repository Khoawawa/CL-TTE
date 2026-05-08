import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.PositionalEncoding import PositionalEncoding1D, CyclicalTimeEncoding
from models.loss.contrastive_loss import HardContrastiveLoss
from models.blocks.cl import MSM
from models.blocks.poi import PoiEncoder, GlobalFiLM
from models.blocks.moco import MoCo

from models.profiler.profiler import BlockTimer

class ContrastiveEncoder(nn.Module):
    def __init__(self, d_model, nhead, tau_I, dropout=0.1, nlayer=4, contrastive_temperature=0.25):
        super().__init__()
        self.tau_I = tau_I
        self.contrastive_temperature = contrastive_temperature
        
        self.moco = MoCo(
            encoder_q= MSM(d_model,nhead,dropout,nlayer),
            encoder_k= MSM(d_model, nhead, dropout, nlayer),
            nemb=d_model,
            nout=d_model,
            queue_size=4096,
            temperature=contrastive_temperature,
            tau_I=tau_I
        )
    
    def forward(self,x,x_aug,src_key_padding_mask=None,src_key_augment_padding_mask=None, y=None):
        # x: (B, T, D)
        # x_aug: (B, T, D)
        
        B, T, D = x.shape
        
        if src_key_padding_mask is None:
            pad_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        else:
            pad_mask = src_key_padding_mask
        
        if src_key_augment_padding_mask is None:
            augment_pad_mask = torch.ones(x_aug.size(0), x_aug.size(1), dtype=torch.bool, device=x_aug.device)
        else:
            augment_pad_mask = src_key_augment_padding_mask
        # encode
        kwargs_q = {"x" : x, "src_key_padding_mask": pad_mask}
        kwargs_k = {"x": x_aug, "src_key_padding_mask": augment_pad_mask}
            
        logits, softweights, h_msm = self.moco(kwargs_q,kwargs_k, y_q=y)
        
        return h_msm, logits, softweights
    
    
class SegmentEncoder(nn.Module):
    def __init__(self,n_poi_groups, nlayers, d_model=128):
        
        super().__init__()
        self.n_poi_groups = n_poi_groups
        
        highway_dim = 6
        week_dim = 4
        date_dim = 10
        time_dim = 48
        poi_dim = 16
        speed_dim = 4
        lanes_dim = 3
        
        self.datetime_dim = week_dim + date_dim + time_dim
        
        self.highwayembed = nn.Embedding(17, highway_dim, padding_idx=0)
        self.speedembed = nn.Embedding(11, speed_dim, padding_idx=0)
        self.lanesembed = nn.Embedding(7, lanes_dim, padding_idx=0)
        # self.gpsembed = nn.Linear(4, gps_dim)
        
        self.weekembed = CyclicalTimeEncoding(d_model=week_dim, period=7)
        self.dateembed = CyclicalTimeEncoding(d_model=date_dim, period=365)
        self.timeembed = CyclicalTimeEncoding(d_model=time_dim, period=1440)
        
        self.poi_embed = PoiEncoder(
            n_poi_groups=n_poi_groups,
            embed_dim=poi_dim
        )
        
        
        modulate_dim = 2 * highway_dim + poi_dim + speed_dim + lanes_dim
        feature_dim = 2 + modulate_dim
        
        
        self.represent = nn.Sequential(
            nn.Linear(feature_dim, d_model)
        )
        
    def forward(self, links, dateinfo, profiler: BlockTimer=None): 
        # this should accomodate both original and augmented
        # links: (2B, T, D_in) 2 * [HighwayID1, HighwayID2, Len, CumLen, Lat1, Lon1, Lat2, Lon2, poi*n, speed, lanes]
        # dateinfo: (B, 3)
        # mask: (2B, T)
        Bx2, T, _ = links.shape
        B = Bx2 // 2
        # print(dateinfo[:4, :])          # are values actually varying?
        # print(dateinfo[:, 2].unique())  # how many distinct times in a batch?
        # since we use the same dateinfo for both original and augmented, we only need to compute it once
        weekrep   = self.weekembed(dateinfo[:B, 0]).squeeze(1)
        daterep   = self.dateembed(dateinfo[:B, 1]).squeeze(1)
        timerep   = self.timeembed(dateinfo[:B, 2]).squeeze(1)
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1) # (B, datetime_dim)
        # spatial features
        highwayrep1 = self.highwayembed(links[:, :, 0].long()) # 6
        highwayrep2 = self.highwayembed(links[:, :, 1].long()) # 6
        highwayrep = torch.cat([highwayrep1, highwayrep2], dim=-1)
        # speed and lanes
        speedrep = self.speedembed(links[:, :, 4].long()) # 4
        lanesrep = self.lanesembed(links[:, :, 5].long()) # 3
        
        # gpsrep = torch.tanh(self.gpsembed(links[:, :, 4:8].float())) # 16
        
        poirep = self.poi_embed(links[:, :, 6:6+self.n_poi_groups]) # (2B, T, poi_dim)
        len_feats = links[:, :, 2:4] # (2B, T, 2)

        features = torch.cat(
            [
                len_feats, # len and cumlen
                highwayrep,
                poirep,
                speedrep,
                lanesrep
            ],
            dim=-1
        )
        features_proj = self.represent(features) # (2B,T,seq_hidden_dim)
        
        return features_proj, datetimerep # (2B, T, seq_hidden_dim), (B, datetime_dim)
    