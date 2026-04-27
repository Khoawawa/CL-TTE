import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.PositionalEncoding import PositionalEncoding1D, PositionalEncodingIndex
from models.loss.contrastive_loss import HardContrastiveLoss
from models.loss.reconstruction_loss import ReconstructionLoss
from models.blocks.cl import MSM
from models.blocks.poi import PoiEncoder, GlobalFiLM

from models.profiler.profiler import BlockTimer

class ContrastiveEncoder(nn.Module):
    def __init__(self, d_model, nhead, r_seconds=45, dropout=0.1, nlayer=4, contrastive_temperature=0.25, recon_weight=0.1):
        super().__init__()
        self.r_seconds = r_seconds
        self.contrastive_temperature = contrastive_temperature
        self.recon_weight = recon_weight

        self.msm = MSM(d_model,nhead,dropout,nlayer)
        
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.loss = HardContrastiveLoss(temperature=self.contrastive_temperature)
        self.recon_loss = ReconstructionLoss(d_model)
        
    def create_pos_mask(self, y_true):
        # y_true: (B, T)
        if y_true.dim() == 2:
            y_true = y_true.squeeze(-1)
            
        y_true = y_true.detach()
        B = y_true.size(0)
        device = y_true.device
        
        dist = torch.abs(y_true.unsqueeze(0) - y_true.unsqueeze(1))
        
        pos_base = (dist <= self.r_seconds).float()   # (B, B)
        pos_base.fill_diagonal_(0)
        # expand to 2B
        pos_mask = torch.zeros(2*B, 2*B, device=y_true.device)

        # ori ↔ ori
        pos_mask[:B, :B] = pos_base

        # aug inherits ori
        pos_mask[B:, :B] = pos_base
        pos_mask[:B, B:] = pos_base

        # DO NOT include aug ↔ aug
        # pos_mask[B:, B:] = 0

        # identity pairs (strong positives)
        idx = torch.arange(B, device=y_true.device)
        pos_mask[idx, idx+B] = 1.0
        pos_mask[idx+B, idx] = 1.0

        return pos_mask
    
    def masked_mean_pooling(self, x, pad_mask):
        # x: (B, T, D)
        # pad_mask: (B, T) with 0 = valid, 1 = padding
        valid_mask = ~pad_mask
        # sum
        x = x * valid_mask.unsqueeze(-1)
        
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        
        return x.sum(dim=1) / denom
    
    def forward(self,x,x_aug,src_key_padding_mask=None,src_key_augment_padding_mask=None,y_true=None):
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
        if self.training:
            x_masked, mask_positions = self.recon_loss.apply_mask(x, pad_mask)
            
            x_all = torch.cat([x_masked,x_aug],dim=0) # (2B, T, D)
            pad_mask_all = torch.cat([pad_mask,augment_pad_mask],dim=0) # (2B, T)
            
            # MSM handle PE
            h_all = self.msm(x_all,pad_mask_all)
            
            z_all = self.masked_mean_pooling(h_all,pad_mask_all) # (2B, D)
            
            pos_mask = self.create_pos_mask(y_true)
            z_all_proj = F.normalize(self.proj(z_all), dim=-1)
            # contrastive learning
            l_cl = self.loss(z_all_proj, pos_mask)
            
            h_msm = h_all[:B]
            l_recon = self.recon_loss(h_msm, x, mask_positions)
            l_combined = l_cl + self.recon_weight * l_recon
            
        else:
            h_msm = self.msm(x,pad_mask)
            
            l_combined = None
        
        return h_msm, l_combined
    
    
class SegmentEncoder(nn.Module):
    def __init__(self,n_poi_groups, nlayers, d_model=128):
        
        super().__init__()
        self.n_poi_groups = n_poi_groups
        
        highway_dim = 6
        week_dim = 3
        date_dim = 10
        time_dim = 20
        poi_dim = 16
        speed_dim = 4
        lanes_dim = 3
        
        self.datetime_dim = week_dim + date_dim + time_dim
        
        self.highwayembed = nn.Embedding(17, highway_dim, padding_idx=0)
        self.speedembed = nn.Embedding(11, speed_dim, padding_idx=0)
        self.lanesembed = nn.Embedding(7, lanes_dim, padding_idx=0)
        # self.gpsembed = nn.Linear(4, gps_dim)
        
        self.weekembed = nn.Embedding(8, week_dim)
        self.dateembed = PositionalEncoding1D(date_dim)
        self.timeembed = PositionalEncoding1D(d_model=time_dim)
        
        self.poi_embed = PoiEncoder(
            n_poi_groups=n_poi_groups,
            embed_dim=poi_dim
        )
        
        
        modulate_dim = 2 * highway_dim + poi_dim + speed_dim + lanes_dim
        feature_dim = 2 + modulate_dim
        
        # film modulator
        self.film = GlobalFiLM(
            time_dim=self.datetime_dim,
            embed_dim=modulate_dim,
            n_layers=nlayers
        )
        
        self.represent = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
        )
        
    def forward(self, links, dateinfo, profiler: BlockTimer=None): 
        # this should accomodate both original and augmented
        # links: (2B, T, D_in) 2 * [HighwayID1, HighwayID2, Len, CumLen, Lat1, Lon1, Lat2, Lon2, poi*n, speed, lanes]
        # dateinfo: (B, 3)
        # mask: (2B, T)
        B, T, _ = links.shape
        
        weekrep   = self.weekembed(dateinfo[:, 0].long())
        daterep   = self.dateembed(dateinfo[:, 1])
        timerep   = self.timeembed(dateinfo[:, 2])
        
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1) # (B, datetime_dim)
        datetimerep_expand = torch.cat([datetimerep, datetimerep], dim=0) # (2B, datetime_dim)
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
        
        modulate_feats = torch.cat(
            [
                highwayrep,
                poirep,
                speedrep,
                lanesrep,
            ],
            dim=-1
        )
        
        # FILM CONDITIONING
        modulate_feats = self.film(modulate_feats,datetimerep_expand)
        
        features = torch.cat(
            [
                len_feats, # len and cumlen
                modulate_feats
            ],
            dim=-1
        )
        features_proj = self.represent(features) # (2B,T,seq_hidden_dim)
        
        return features_proj, datetimerep
    