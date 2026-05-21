import torch
import torch.nn as nn
import torch.nn.functional as F
from models.contrastive_module import ContrastiveModule
from models.blocks.LayerNormGRU import LayerNormGRU
from models.blocks.poi import GlobalFiLM

from models.profiler.profiler import BlockTimer
    
class Cl_TTE(nn.Module):
    def __init__(self, d_model, nhead, seq_layer, gru_layers, tau_I,n_poi_groups=9,contrastive_temperature=0.25, use_contrastive=False):
        super().__init__()
        
        self.d_model = d_model
        
        self.contrastive_module = ContrastiveModule(
            d_model=d_model,
            nhead=nhead,
            tau_I=tau_I,
            dropout=0.1,
            nlayer=seq_layer,
            n_poi_groups=n_poi_groups,
            contrastive_temperature=contrastive_temperature
        )
        
        self.temporal_block = LayerNormGRU(input_dim=d_model, hidden_dim=d_model, num_layers=gru_layers)
        
        # ATTENTION POOLING
        self.pool_query = nn.Parameter(torch.randn(1,1,d_model))
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        
        # REGRESSION
        mlp_in_dim = d_model + self.contrastive_module.time_encoder.datetime_dim
        self.pre_regression_norm = nn.LayerNorm(d_model)
        self.regression_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim//2, 1)
        )
        
    def forward(self, inputs: torch.Tensor, y_true: torch.Tensor, profiler: BlockTimer=None):
        # inputs: 
        # links:
        # dateinfo : [B, 3]
        # culm_len: [B, T, 1]
        # lens: [B]
        x = inputs['links_clean']
        x_aug = inputs['links_aug']
        dateinfo = inputs['dateinfo']
        culm_len = inputs['culm_len']
        link_index = inputs['link_index']
        lens = inputs['lens']
        
        max_len = torch.max(lens).item()
        segment_mask = torch.arange(max_len, device=lens.device).unsqueeze(0) < lens.unsqueeze(1)
        padding_mask = ~segment_mask
        
        h_msm, datetimerep, loss_cl, metric = self.contrastive_module(
            x, x_aug, dateinfo, culm_len, link_index,
            src_key_padding_mask=padding_mask, 
            src_key_augment_padding_mask=padding_mask
        )

        gru_input = h_msm.transpose(0,1).contiguous() # (T, B, D)
        h,_ = self.temporal_block(gru_input, lens) # (B, T, D)
        h = h.transpose(0,1).contiguous()
        
        # ATTENTION POOLING
        if profiler: profiler.start('attn pooling')
        query = self.pool_query.expand(h.size(0), -1, -1) # (B, 1, D)
        z = self.attn(query, h, h, key_padding_mask=padding_mask)[0].squeeze(1) # (B, D)
        if profiler: profiler.stop()
        
        # REGRESSION
        z = self.pre_regression_norm(z)
        z_time = torch.concat([z, datetimerep], dim=-1) # (B,D + 33)
        t = self.regression_mlp(z_time) # (B, 1)
        
        return t, loss_cl, metric
    
    