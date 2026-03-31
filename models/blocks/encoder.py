import math

import torch
import torch.nn as nn
from models.base.PositionalEncoding import CyclicalTimeEncoding, PositionalEncoding1D
from models.blocks.cl import MSM, ReCo

class FullEncoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, nlayer=4):
        super().__init__()
        self.middle_manager = MiddleManager(d_model, nhead, dropout, nlayer)
        self.pad_token = nn.Parameter(torch.zeros(1, 1, d_model))
    def apply_merge(self,x, start_mask, pad_mask, pad_token):

        B, T, _ = x.shape

        span_mask = start_mask | pad_mask
        span_mask_f = span_mask.float().unsqueeze(-1)

        summed = (x * span_mask_f).sum(dim=1)
        counts = span_mask_f.sum(dim=1).clamp(min=1e-6)

        merged = summed / counts

        merged_expand = merged.unsqueeze(1).expand(-1, T, -1)

        x_aug = torch.where(start_mask.unsqueeze(-1), merged_expand, x)
        x_aug = torch.where(pad_mask.unsqueeze(-1), pad_token, x_aug)

        return x_aug
    
    def masked_mean_pooling(self, x, mask):
        # x: (B, T, D)
        # mask: (B, T) with 1 = valid, 0 = padding
        mask = mask.float()
        # sum
        x_sum = torch.einsum('btd,bt->bd', x, mask)
        # count valid tokens
        denom = mask.sum(dim=1, keepdim=True)  # (B, 1)
        # avoid division by zero
        denom = denom.clamp(min=1e-4)
        return x_sum / denom
    def check_positive(self, y_true, r_percentile=0.2):
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
    def augment(self, x, strategy: str, **kwargs):
        if strategy == "merge":
            return self.apply_merge(x, **kwargs)
        else:
            raise NotImplementedError
    def contrastive_loss(self, z, y_true, r):
        # 2 positive cases: 
        # - same sequence, diff augmentation 
        # - label in the same ball#
        B_2, D = z.shape
        B = B_2 // 2
        pos_mask = self.check_positive(y_true, r)
        
        
    def forward(self, x, y_true, **kwargs):
        x_aug = self.augment(x, 'merge', **kwargs)
        
        h = self.middle_manager(x, **kwargs)
        h_aug = self.middle_manager(x_aug, **kwargs)
        
        z = self.masked_mean_pooling(h, kwargs['valid_mask'])
        z_aug = self.masked_mean_pooling(h_aug, kwargs['valid_mask'])
        
        # F.normalize is now handled safely inside contrastive_loss!
        z_all = torch.cat([z,z_aug],dim=0) # (2B, D)
        
        l_cl = self.contrastive_loss(z_all, y_true)
        
        return h, l_cl
        
        
class MiddleManager(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, nlayer=4):
        super().__init__()
        self.seg_enc = SegmentEncoder(d_model=d_model)
        self.reco = ContrastiveEncoder(d_model=d_model, nhead=nhead, dropout=dropout, nlayer=nlayer)
    def forward(self, x, **kwargs):
        x = self.seg_enc(x, kwargs['linkinfo'], kwargs['dateinfo'])
        x = self.reco(x, kwargs['lens'])
        return x
    
class ContrastiveEncoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, nlayer=4):
        super().__init__()
        self.reco = MSM(d_model,nhead,dropout,nlayer)
    
    def forward(self, x, lens):
        B, T, D = x.shape
        device = x.device
        
        src_padding_mask = torch.arange(T, device=device).unsqueeze(0) >= lens.unsqueeze(1)  # (B, T)
        return self.reco(x, src_padding_mask, lens)
    
class SegmentEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()

        self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        self.gpsembed = nn.Linear(4,16)
        
        self.weekembed = nn.Embedding(8, 3)
        self.dateembed = PositionalEncoding1D(10)
        self.timeembed = PositionalEncoding1D(d_model=20)
        
        mlp_in_dim = 2 + 5 + 16 + 3 + 10 + 20 # = 56 // 8
        
        self.datetime_dim = 3 + 10 + 20
        
        self.represent = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_in_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim * 2, d_model)
        )
        
    def apply_segment_mask(self, x, start_mask, pad_mask, mask_token):
        """
        x: (B, T, D)
        start_mask: (B, T) -> True where segment should be masked
        pad_mask: (B, T) -> True where padding
        mask_token: (D,) or (1, 1, D)
        """
        B, T, D = x.shape

        x_aug = x.clone()

        # ensure mask_token shape is broadcastable
        if mask_token.dim() == 1:
            mask_token = mask_token.view(1, 1, D)

        # apply segment masking (hide information, keep structure)
        x_aug = torch.where(start_mask.unsqueeze(-1), mask_token, x_aug)

        # keep padding behavior unchanged
        x_aug = torch.where(pad_mask.unsqueeze(-1), mask_token, x_aug)

        return x_aug
    
    def forward(self, links, dateinfo):
        # links: (B, T, 7)
        # dateinfo: (B, 3)
        # lens: (B,)
        B, T, _ = links.shape
        
        weekrep   = self.weekembed(dateinfo[:, 0].long())
        daterep   = self.dateembed(dateinfo[:, 1])
        timerep   = self.timeembed(dateinfo[:, 2])
        
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1)
        datetimerep_expand = datetimerep.unsqueeze(1).expand(-1,T, -1) # (B,T,seq_hidden_dim)
        # spatial features
        highwayrep = self.highwayembed(links[:, :, 0].long()) # 5
        
        gpsrep = torch.tanh(self.gpsembed(links[:, :, 3:7].float())) # 16
        
        features = torch.cat([links[..., 1:3], gpsrep,highwayrep, datetimerep_expand], dim=-1) # 2 + 5 + 16 + 33 
        
        features_proj = self.represent(features) # (B,T,seq_hidden_dim)
        
        return features_proj, datetimerep
    