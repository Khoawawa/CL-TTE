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
        
    def calculate_contrastive_loss(
        self,
        z1,
        z2,
        ignore_mask
    ):

        B = z1.size(0)

        # ---------------------------------------------
        # projector
        # ---------------------------------------------

        z1_raw = z1.float()
        z2_raw = z2.float()

        z1_proj = self.projector(z1_raw)
        z2_proj = self.projector(z2_raw)

        z1 = F.normalize(
            z1_proj.float(),
            dim=-1
        )

        z2 = F.normalize(
            z2_proj.float(),
            dim=-1
        )

        with torch.amp.autocast(
            device_type='cuda',
            enabled=False
        ):

            # =========================================
            # sim matrix
            # =========================================
            raw_sim = torch.matmul(z1, z2.T)
            
            sim = raw_sim / self.contrastive_temperature

            logits_mask = ~torch.eye(
                B,
                dtype=torch.bool,
                device=sim.device
            )

            # =========================================
            # positives are ONLY paired augmentations
            # diagonal = positive
            # =========================================

            pos_mask = torch.eye(
                B,
                device=sim.device
            )
            
                
            if ignore_mask is not None:
                masked_sim = sim.masked_fill(
                    ignore_mask | pos_mask.bool(),
                    -1e9
                )
            else:
                with torch.no_grad():
                    false_negative_mask = raw_sim > 0.95
                    false_negative_mask.fill_diagonal_(False)
                masked_sim = sim.masked_fill(
                    false_negative_mask | pos_mask.bool(),
                    -1e9
                )

            # =========================================
            # InfoNCE
            # =========================================

            log_denom = torch.logsumexp(
                masked_sim,
                dim=1,
                keepdim=True
            )

            log_prob = sim - log_denom

            loss = -(
                pos_mask * log_prob
            ).sum(dim=1)

            loss = loss.mean()

            # =========================================
            # anti-collapse
            # =========================================

            all_repr = torch.cat(
                [z1_raw, z2_raw],
                dim=0
            )

            loss_var = self.variance_loss(
                all_repr
            )

            loss_cov = self.covariance_loss(
                all_repr
            )

            total_loss = (
                loss +
                1.0 * loss_var +
                0.1 * loss_cov
            )

        # =====================================================
        # metrics
        # =====================================================

        with torch.no_grad():

            offdiag = raw_sim[logits_mask]
            
            metric = {
                'embed_std_raw':
                    all_repr.std(dim=0).mean().item(),

                'embed_std_proj':
                    torch.cat(
                        [z1_proj, z2_proj],
                        dim=0
                    ).std(dim=0).mean().item(),

                'offdiag_mean':
                    offdiag.mean().item(),

                'offdiag_std':
                    offdiag.std().item(),

                'offdiag_min':
                    offdiag.min().item(),

                'offdiag_max':
                    offdiag.max().item(),

                'loss_var':
                    loss_var.item(),

                'loss_cov':
                    loss_cov.item(),
            }
            if ignore_mask is not None:
                mask_ratio = false_negative_mask.float().mean().item()
                metric['mask_ratio'] = mask_ratio
                mask_sum = ignore_mask.float().sum().item()
                metric['mask_sum'] = mask_sum

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
    def encode(
        self,
        x,
        src_key_padding_mask=None
    ):

        B, T, D = x.shape

        if src_key_padding_mask is None:

            pad_mask = torch.zeros(
                B,
                T,
                dtype=torch.bool,
                device=x.device
            )

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

        return h, trip_repr
        
    def forward(
        self,
        orig_repr,
        aug_repr,
        ignore_mask,
        src_key_padding_mask=None,
        src_key_augment_padding_mask=None
    ):
        # x: (B, T, D)
        
        h_orig, trip_repr_orig = self.encode(
            orig_repr,
            src_key_padding_mask=src_key_padding_mask
        )
        loss_cl = None
        metric = None
        if self.training:
            _, trip_repr_aug = self.encode(
                aug_repr,
                src_key_padding_mask=src_key_augment_padding_mask
            )
            
            
            loss_cl, metric = self.calculate_contrastive_loss(
                trip_repr_orig,
                trip_repr_aug,
                ignore_mask
            )
        

        return h_orig, loss_cl, metric
    
class TimeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        week_dim = 4
        date_dim = 10
        time_dim = 48
        
        self.datetime_dim = week_dim + date_dim + time_dim
        
        self.weekembed = CyclicalTimeEncoding(d_model=week_dim, period=7)
        self.dateembed = CyclicalTimeEncoding(d_model=date_dim, period=365)
        self.timeembed = CyclicalTimeEncoding(d_model=time_dim, period=1440)
    
    def forward(self, dateinfo):
        weekrep   = self.weekembed(dateinfo[:, 0]).squeeze(1)
        daterep   = self.dateembed(dateinfo[:, 1]).squeeze(1)
        timerep   = self.timeembed(dateinfo[:, 2]).squeeze(1)
        
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1) # (B, datetime_dim)
        
        return datetimerep
    
class SegmentEncoder(nn.Module):
    def __init__(self,n_poi_groups, d_model=128):
        
        super().__init__()
        self.n_poi_groups = n_poi_groups
        
        highway_dim = 6
        poi_dim = 16
        speed_dim = 4
        lanes_dim = 3
        
        
        self.highwayembed = nn.Embedding(17, highway_dim, padding_idx=0)
        self.speedembed = nn.Embedding(13, speed_dim, padding_idx=0)
        # self.lanesembed = nn.Embedding(7, lanes_dim, padding_idx=0)
        # self.gpsembed = nn.Linear(4, gps_dim)
        
        self.poi_embed = PoiEncoder(
            n_poi_groups=n_poi_groups,
            embed_dim=poi_dim
        )
        self.feature_dim = 2 * highway_dim + poi_dim + speed_dim + 1
        
        
        # self.represent = nn.Sequential(
        #     nn.Linear(feature_dim, d_model)
        # )
        
    def forward(self, links): 
        # links: (B, T, D_in) [HighwayID1, HighwayID2, Len, CumLen, Lat1, Lon1, Lat2, Lon2, poi*n, speed, lanes]
        # dateinfo: (B, 3)
            
        # spatial features
        highwayrep1 = self.highwayembed(links[:, :, 0].long()) # 6
        highwayrep2 = self.highwayembed(links[:, :, 1].long()) # 6
        highwayrep = torch.cat([highwayrep1, highwayrep2], dim=-1)
        
        # speed and lanes
        speedrep = self.speedembed(links[:, :, 3].long()) # 4
        # lanesrep = self.lanesembed(links[:, :, 5].long()) # 3
        
        # gpsrep = torch.tanh(self.gpsembed(links[:, :, 4:8].float())) # 16
        
        poirep = self.poi_embed(links[:, :, 4:4+self.n_poi_groups]) # (B, T, poi_dim)

        semantic_feats = torch.cat(
            [
                links[:,:,2].unsqueeze(-1), # len
                highwayrep,
                poirep,
                speedrep,
                # lanesrep
            ],
            dim=-1
        )
        # semantic_feats = self.represent(semantic_feats) # (B, T, d_model)
        return semantic_feats # (B, T, d_model), (B, T, 2)
    