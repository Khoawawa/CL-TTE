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
        
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.xavier_uniform_(self.cls_token)
    
    def calculate_contrastive_loss(self, z_orig, z_aug, y_true):
        B = z_orig.size(0)
        
        z_orig = F.normalize(z_orig, dim=-1)
        z_aug = F.normalize(z_aug, dim=-1)
        
        # FIX 1: Measure actual feature spread, not norm spread
        print(f"z_orig std: {z_orig.std(dim=0).mean():.4f}")
        print(f"z_aug  std: {z_aug.std(dim=0).mean():.4f}")
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            z_orig = z_orig.float()
            z_aug = z_aug.float()
            
            sim = torch.matmul(z_orig, z_aug.T) 
        
            # FIX 2: Safely extract off-diagonal mean without zero-bias
            mask = ~torch.eye(B, dtype=torch.bool, device=z_orig.device)
            print(f"sim diag mean: {torch.diag(sim).mean():.4f}")   
            print(f"sim offdiag mean: {sim[mask].mean():.4f}") 
            
            sim = sim / self.contrastive_temperature
            sim = sim.clamp(-100, 100)  
            
            log_probs = F.log_softmax(sim, dim=-1)
            l_pos = -torch.diag(log_probs)
            
            if y_true is None:
                labels = torch.arange(B, device=z_orig.device)
                loss = F.cross_entropy(sim, labels)
                return torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
        
            # Soft negative contrastive loss
            y_true = y_true.squeeze(-1).detach().float()
        
            time_diff = torch.abs(y_true.unsqueeze(1) - y_true.unsqueeze(0))
            soft_weights = 2 * torch.sigmoid(-self.tau_I * time_diff)
            
            soft_weights = soft_weights * mask.float() 
        
            weight_sum = soft_weights.sum(dim=1).clamp(min=1e-6)
            
            # FIX 3: Remove the unsqueeze to prevent [B, B] broadcasting explosion!
            l_neg = -(soft_weights * log_probs).sum(dim=1) / weight_sum
            
        loss = (l_pos + l_neg).mean()
        print(f"Contrastive Loss: {loss.item():.4f} (pos: {l_pos.mean().item():.4f}, neg: {l_neg.mean().item():.4f})")
        
        return torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
        
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
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        if src_key_padding_mask is None:
            pad_mask = torch.ones(B, T + 1, dtype=torch.bool, device=x.device)
        else:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            pad_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
            
        x_clean = self.input_proj(x)
        x_clean_with_cls = torch.cat([cls_tokens, x_clean], dim=1)
        
                
        h_msm_full = self.transformer(x_clean_with_cls, src_key_padding_mask=pad_mask, use_heavy_dropout=False)
        loss_cl = None
        
        if self.training and use_contrastive:
            
            x_aug = F.dropout(x_clean, p=0.15, training=True)
            x_aug = self.input_proj(x_aug)
            x_aug_with_cls = torch.cat([cls_tokens, x_aug], dim=1)
            h_msm_aug = self.transformer(x_aug_with_cls, src_key_padding_mask=pad_mask, use_heavy_dropout=False)
            
            print(f"h_clean vs h_aug cosine: {F.cosine_similarity(h_msm_full.mean(1), h_msm_aug.mean(1)).mean():.4f}")

            h_cls_orig = h_msm_full[:, 0, :]
            h_cls_aug = h_msm_aug[:, 0, :]
            
            z_clean = self.projector(h_cls_orig)
            z_aug = self.projector(h_cls_aug)
            
            loss_cl = self.calculate_contrastive_loss(z_clean, z_aug, y)
        
        h_msm_seq = h_msm_full[:, 1:, :]
        
        return h_msm_seq, loss_cl
    
    
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
        len_feats = links[:, :, 2:4] # (B, T, 2)

        semantic_feats = torch.cat(
            [
                highwayrep,
                poirep,
                speedrep,
                lanesrep
            ],
            dim=-1
        )
        # semantic_feats = self.represent(semantic_feats) # (B, T, d_model)
        return semantic_feats, len_feats, datetimerep # (B, T, d_model), (B, T, 2), (B, datetime_dim)
    