import torch
import torch.nn as nn

from models.blocks.encoder import ContrastiveEncoder, SegmentEncoder, TimeEncoder
from models.blocks.poi import GlobalFiLM


class ContrastiveModule(nn.Module):
    def __init__(self, d_model, nhead, tau_I, dropout=0.1, nlayer=4, n_poi_groups=9,contrastive_temperature=0.25):
        super().__init__()
        self.tau_I = tau_I
        self.contrastive_temperature = contrastive_temperature
        
        self.segment_encoder = SegmentEncoder(n_poi_groups=n_poi_groups, d_model=d_model)
        
        self.time_encoder = TimeEncoder()
        
        self.film = GlobalFiLM(
            time_dim = self.time_encoder.datetime_dim,
            embed_dim = self.segment_encoder.feature_dim,
            n_layers = nlayer // 2
        )
        
        self.contrastive_encoder = ContrastiveEncoder(
            in_dim=self.segment_encoder.feature_dim, 
            d_model=d_model, 
            nhead=nhead,
            tau_I=tau_I, 
            dropout1=dropout, 
            nlayer=nlayer, 
            contrastive_temperature=contrastive_temperature
        )
        
        self.after_proj = nn.Linear(d_model + 1, d_model)
    
    def forward(self, x, x_aug, dateinfo, culm_len, ignore_mask, src_key_padding_mask=None, src_key_augment_padding_mask=None):
        # x: (B, T, D)
        # x_aug: (B, T, D)
        # dateinfo: (B, 3)
        # culm_len: (B, T, 1)
        
        datetimerep = self.time_encoder(dateinfo)
        
        orig_repr = self.segment_encoder(x)
        aug_repr = self.segment_encoder(x_aug)
        
        orig_repr = self.film(orig_repr, datetimerep)
        aug_repr = self.film(aug_repr, datetimerep)
                
        h_msm, loss_cl, metric = self.contrastive_encoder(
            orig_repr,
            aug_repr,
            ignore_mask=ignore_mask,
            src_key_padding_mask=src_key_padding_mask,
            src_key_augment_padding_mask=src_key_augment_padding_mask
        )
        
        h_msm = self.after_proj(torch.cat([h_msm, culm_len], dim=-1))
        
        return h_msm.detach(), datetimerep,loss_cl, metric