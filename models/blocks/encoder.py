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
    def __init__(self, in_dim, d_model, nhead, tau_I, dropout1=0.1, dropout2=0.3, nlayer=4, contrastive_temperature=0.25):
        super().__init__()
        self.tau_I = tau_I
        self.contrastive_temperature = contrastive_temperature
        
        
        self.input_proj = nn.Linear(in_dim, d_model)
        
        self.transformer = MSM(d_model,nhead,dropout1,dropout2,nlayer)
        self.projector = Projector(d_model, d_model)
        
    def calculate_contrastive_loss(self, z, y_true):
        B = z.size(0)

        z = F.normalize(z.float(), dim=-1)

        with torch.amp.autocast(device_type='cuda', enabled=False):

            sim = torch.matmul(z, z.T)
            sim = sim / self.contrastive_temperature

            logits_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)

            y = y_true.squeeze(-1).float()
            t1 = y.unsqueeze(1)
            t2 = y.unsqueeze(0)

            rel_diff = torch.abs(t1 - t2) / (torch.max(t1, t2) + 1e-6)

            # Three zones
            pos_mask = (rel_diff <= 0.02) & logits_mask        # pull together
            neg_mask = (rel_diff > 0.25) & logits_mask         # push apart
            # ignore zone: 0.03 < rel_diff <= 0.15 — ambiguous, don't touch

            sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
            
            alpha_neg = (pos_mask.sum() * 5 / neg_mask.sum().clamp(min=1)).item()

            exp_sim_pos = torch.exp(sim) * pos_mask.float()
            exp_sim_neg = torch.exp(sim) * neg_mask.float() * alpha_neg
            denom = (exp_sim_pos + exp_sim_neg).sum(dim=1, keepdim=True)
            log_prob = sim - torch.log(denom.clamp(min=1e-8))

            pos_count = pos_mask.sum(dim=1)
            valid_rows = pos_count > 0

            loss = -(
                (pos_mask.float() * log_prob).sum(dim=1)
                / pos_count.clamp(min=1)
            )
            loss = loss[valid_rows].mean()

        with torch.no_grad():
            raw_sim = torch.matmul(z, z.T)
            offdiag_sims = raw_sim[logits_mask]

            metric = {
                'diag': torch.diag(raw_sim).mean().item(),
                'offdiag_mean': offdiag_sims.mean().item(),
                'offdiag_std': offdiag_sims.std().item(),
                'offdiag_min': offdiag_sims.min().item(),
                'offdiag_max': offdiag_sims.max().item(),
                'pos': pos_mask.sum().item(),
                'neg': neg_mask.sum().item(),           # ← new
                'dampened_neg': (neg_mask.sum() * alpha_neg).item(),
                'valid_rows': valid_rows.sum().item(),  # ← new
            }

        return loss, metric

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
    def forward(self,x,src_key_padding_mask=None, y=None, use_contrastive=False):
        # x: (B, T, D)
        
        B, T, D = x.shape
        
        if src_key_padding_mask is None:
            pad_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        else:
            pad_mask = src_key_padding_mask
            
        x_clean = self.input_proj(x)
        
                
        h_msm_full = self.transformer(x_clean, src_key_padding_mask=pad_mask, use_heavy_dropout=False)
        loss_cl = None
        metric = None
        if self.training and use_contrastive:
            
            h_cls_orig = self.masked_mean_pool(h_msm_full, padding_mask=pad_mask)               
            z_clean = self.projector(h_cls_orig)
            
            loss_cl, metric = self.calculate_contrastive_loss(z_clean, y)
        
        
        return h_msm_full, loss_cl, metric
    
    
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
        
        
        modulate_dim = 2 * highway_dim + poi_dim + speed_dim + lanes_dim + 1
        self.feature_dim = modulate_dim
        
        
        # self.represent = nn.Sequential(
        #     nn.Linear(feature_dim, d_model)
        # )
        
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
        culm_len = links[:, :, 3].unsqueeze(-1) # (B, T, 1)

        semantic_feats = torch.cat(
            [
                links[:,:,2].unsqueeze(-1), # len
                highwayrep,
                poirep,
                speedrep,
                lanesrep
            ],
            dim=-1
        )
        # semantic_feats = self.represent(semantic_feats) # (B, T, d_model)
        return semantic_feats, culm_len, datetimerep # (B, T, d_model), (B, T, 2), (B, datetime_dim)
    