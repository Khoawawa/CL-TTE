import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.PositionalEncoding import PositionalEncoding1D, CyclicalTimeEncoding
from models.loss.contrastive_loss import HardContrastiveLoss
from models.blocks.cl import MSM
from models.blocks.poi import PoiEncoder, GlobalFiLM
from models.blocks.moco import MoCo, Projector

from models.profiler.profiler import BlockTimer

class ContrastiveEncoder(nn.Module):
    def __init__(self, d_model, nhead, tau_I, dropout1=0.1, dropout2=0.3, nlayer=4, contrastive_temperature=0.25):
        super().__init__()
        self.tau_I = tau_I
        self.contrastive_temperature = contrastive_temperature
        self.transformer = MSM(d_model,nhead,dropout1,dropout2,nlayer)
        self.projector = Projector(d_model, d_model)
    
    def calculate_contrastive_loss(self, z_orig, z_aug, y_true):
        B = z_orig.size(0)
        
        z_orig = F.normalize(z_orig, dim=-1)
        z_aug = F.normalize(z_aug, dim=-1)
        
        sim = torch.matmul(z_orig, z_aug.T) / self.contrastive_temperature
        
        log_probs = F.log_softmax(sim, dim=-1)
        
        l_pos = -torch.diag(log_probs)
        
        y_true = y_true.squeeze(-1).detach().float()
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            time_diff = torch.abs(y_true.unsqueeze(1) - y_true.unsqueeze(0))
            soft_weights = 2 * torch.sigmoid(-self.tau_I * time_diff)
        print(soft_weights.mean())
        mask = ~torch.eye(B, dtype=torch.bool, device=z_orig.device)
        
        l_neg = -(soft_weights * log_probs * mask).sum(dim=1) / (B - 1 + 1e-6)
        
        loss = (l_pos + l_neg).mean()
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
        
        return loss
        
    def masked_mean_pool(self, x, padding_mask=None):
        """
        x : (B, T, D)
        padding_mask : (B, T)  True = padded

        returns
        pooled : (B, D)
        """
        if padding_mask is None:
            return x.mean(dim=1)

        valid_mask = ~padding_mask
        valid_mask = valid_mask.unsqueeze(-1).float()

        summed = (x * valid_mask).sum(dim=1)
        counts = valid_mask.sum(dim=1).clamp(min=1e-6)

        return summed / counts
    def forward(self,x,src_key_padding_mask=None, y=None):
        # x: (B, T, D)
        
        B, T, D = x.shape
        
        if src_key_padding_mask is None:
            pad_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        else:
            pad_mask = src_key_padding_mask
        
        x_aug = x.clone()
        
        h_msm = self.transformer(x, src_key_padding_mask=pad_mask, use_heavy_dropout=False)
        loss_cl = None
        
        if self.training:
            h_msm_aug = self.transformer(x_aug, src_key_padding_mask=pad_mask, use_heavy_dropout=True)
            
            noise = torch.rand_like(h_msm_aug) * 0.05
            h_msm_aug = h_msm_aug + noise
            
            z_clean = self.projector(self.masked_mean_pool(h_msm, pad_mask))
            z_aug = self.projector(self.masked_mean_pool(h_msm_aug, pad_mask))
            
            loss_cl = self.calculate_contrastive_loss(z_clean, z_aug, y)
        
        return h_msm, loss_cl
    
    
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
        # links: (B, T, D_in) [HighwayID1, HighwayID2, Len, CumLen, Lat1, Lon1, Lat2, Lon2, poi*n, speed, lanes]
        # dateinfo: (B, 3)
        
        weekrep   = self.weekembed(dateinfo[:, 0]).squeeze(1)
        daterep   = self.dateembed(dateinfo[:, 1]).squeeze(1)
        timerep   = self.timeembed(dateinfo[:, 2]).squeeze(1)
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1) # (B, datetime_dim)
        # spatial features
        highwayrep1 = self.highwayembed(links[:, :, 0].long()) # 6
        highwayrep2 = self.highwayembed(links[:, :, 1].long()) # 6
        highwayrep = torch.cat([highwayrep1, highwayrep2], dim=-1)
        # speed and lanes
        speedrep = self.speedembed(links[:, :, 4].long()) # 4
        lanesrep = self.lanesembed(links[:, :, 5].long()) # 3
        
        # gpsrep = torch.tanh(self.gpsembed(links[:, :, 4:8].float())) # 16
        
        poirep = self.poi_embed(links[:, :, 6:6+self.n_poi_groups]) # (B, T, poi_dim)
        len_feats = links[:, :, 2:4] # (B, T, 2)

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
        features_proj = self.represent(features) # (B,T,seq_hidden_dim)
        
        return features_proj, datetimerep # (B, T, seq_hidden_dim), (B, datetime_dim)
    