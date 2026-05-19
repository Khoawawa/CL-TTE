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
        trip_repr_raw = z.float()
        z_proj = self.projector(trip_repr_raw)
        z = F.normalize(z_proj.float(), dim=-1)
        

        with torch.amp.autocast(device_type='cuda', enabled=False):

            sim = torch.matmul(z, z.T)
            sim = sim / self.contrastive_temperature

            logits_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)

            y = y_true.squeeze(-1).float()
            
            t1 = y.unsqueeze(1)
            t2 = y.unsqueeze(0)

            rel_diff = torch.abs(t1 - t2) / (torch.max(t1, t2) + 1e-6)

            target_sim = torch.exp(
                -rel_diff / self.tau_I
            )

            target_sim = target_sim * logits_mask.float()
            
            neg_mask = (
                (rel_diff > 0.15) &
                (rel_diff < 0.9) &
                logits_mask
            )

            sim = sim - sim.max(
                dim=1,
                keepdim=True
            )[0].detach()

            exp_sim = torch.exp(sim)

            pos_term = exp_sim * target_sim
            neg_term = exp_sim * neg_mask.float()

            denom = (
                pos_term.sum(dim=1, keepdim=True) +
                neg_term.sum(dim=1, keepdim=True)
            ).clamp(min=1e-6)

            log_prob = sim - torch.log(denom)

            pos_weight_sum = target_sim.sum(dim=1)

            valid_rows = pos_weight_sum > 1e-6

            loss = -(
                (target_sim * log_prob).sum(dim=1)
                / pos_weight_sum.clamp(min=1e-6)
            )

            loss = loss[valid_rows].mean()
            
            loss_var = self.variance_loss(trip_repr_raw.float())
            loss_cov = self.covariance_loss(trip_repr_raw.float())

            total_loss = loss + 1.0 * loss_var + 0.05 * loss_cov

        with torch.no_grad():

            raw_sim = torch.matmul(z, z.T)

            offdiag = raw_sim[logits_mask]

            metric = {
                'embed_std': z_raw.std(dim=0).mean().item(),
                'offdiag_mean': offdiag.mean().item(),
                'offdiag_std': offdiag.std().item(),
                'offdiag_min': offdiag.min().item(),
                'offdiag_max': offdiag.max().item(),
                'valid_rows': valid_rows.sum().item(),
                'loss_var': loss_var.item(),
                'loss_cov': loss_cov.item(),
            }

        return total_loss, metric

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
    
    def variance_loss(self, z):
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1.0 - std_z))
    
    def covariance_loss(self, z):

        z = z - z.mean(dim=0)

        N, D = z.size()

        cov = (z.T @ z) / (N - 1)

        off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:,1:].flatten()

        return (off_diag ** 2).mean()
    
    def forward(self,x,src_key_padding_mask=None, y=None, use_contrastive=False):
        # x: (B, T, D)
        
        B, T, D = x.shape
        
        if src_key_padding_mask is None:
            pad_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        else:
            pad_mask = src_key_padding_mask
            
        x = self.input_proj(x)
        
        h = self.transformer(
            x,
            src_key_padding_mask=pad_mask
        )
        
        trip_repr = self.masked_mean_pool(
            h,
            pad_mask
        )
        loss_cl = None
        metric = {}

        if use_contrastive and y is not None:

            loss_cl, metric = self.calculate_contrastive_loss(
                trip_repr,
                y
            )

        return h, trip_repr, loss_cl, metric
    
    
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
    