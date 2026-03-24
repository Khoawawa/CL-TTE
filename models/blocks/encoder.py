import math

import torch
import torch.nn as nn
from models.base.PositionalEncoding import CyclicalTimeEncoding, PositionalEncoding1D
from models.blocks.cl import MSM, ReCo

class ContrastiveEncoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, nlayer=4):
        super().__init__()
        self.reco = ReCo(d_model,nhead,dropout,nlayer)
        self.pad_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
    def apply_token_mask(self, x, lens, mask_prob=0.15):
        B, T, D = x.shape
        device = x.device

        valid_mask = torch.arange(T, device=device).unsqueeze(0) < lens.unsqueeze(1)
        rand = torch.rand(B, T, device=device)

        mask = (rand < mask_prob) & valid_mask

        # split into 80/10/10
        rand2 = torch.rand(B, T, device=device)

        mask_token = self.pad_token.expand(B, T, -1).to(dtype=x.dtype)

        x_masked = x.clone()

        # 80% → mask token
        mask_token_mask = mask & (rand2 < 0.8)
        x_masked[mask_token_mask] = mask_token[mask_token_mask]

        # 10% → random token
        random_mask = mask & (rand2 >= 0.8) & (rand2 < 0.9)
        random_indices = torch.randint(0, T, (B, T), device=device)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
        x_masked[random_mask] = x[batch_idx[random_mask], random_indices[random_mask]]
        # 10% → unchanged (do nothing)

        return x_masked, mask
    
    def forward(self, x, lens, mask_prob, noise,r,y_true=None):
        B, T, D = x.shape
        device = x.device
        
        src_padding_mask = torch.arange(T, device=device).unsqueeze(0) >= lens.unsqueeze(1)  # (B, T)
        
        x_aug, _ = self.apply_token_mask(x, lens,mask_prob)
        x_aug = x_aug + torch.randn_like(x_aug) * noise
        
        z, l_cl = self.reco(x,x_aug,r,src_padding_mask,y_true)
        
        return z, l_cl
    
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
    
    def forward(self, links, dateinfo, lens):
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
    