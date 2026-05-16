import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.encoder import SegmentEncoder, ContrastiveEncoder
from models.blocks.LayerNormGRU import LayerNormGRU
from models.blocks.poi import GlobalFiLM

from models.profiler.profiler import BlockTimer
    
class Cl_TTE(nn.Module):
    def __init__(self, d_model, nhead, seq_layer, tau_I,n_poi_groups=9,contrastive_temperature=0.25, use_contrastive=False):
        super().__init__()
        
        self.d_model = d_model
        self.dropout1 = 0.1
        self.dropout2 = 0.1
        self.use_contrastive = use_contrastive
        
        # SEGMENT ENCODER
        self.enc = SegmentEncoder(d_model=d_model, n_poi_groups=n_poi_groups, nlayers=seq_layer)
        
        # CONTRASTIVE BLOCK
        self.contrast_enc = ContrastiveEncoder(in_dim=self.enc.feature_dim, d_model=d_model, nlayer=seq_layer, nhead=nhead, contrastive_temperature=contrastive_temperature, tau_I=tau_I)
        
        # film modulator
        self.film = GlobalFiLM(
            time_dim=self.enc.datetime_dim,
            embed_dim=d_model,
            n_layers=seq_layer
        )
        # self.pre_gru_proj = nn.Linear(d_model + 2, d_model)
        self.temporal_block = LayerNormGRU(input_dim=d_model + 2, hidden_dim=d_model, num_layers=seq_layer)
        
        # ATTENTION POOLING
        self.pool_query = nn.Parameter(torch.randn(1,1,d_model))
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        
        # REGRESSION
        mlp_in_dim = d_model + self.enc.datetime_dim
        self.pre_regression_norm = nn.LayerNorm(d_model)
        self.regression_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim//2, 1)
        )
        
    def forward(self, inputs: torch.Tensor, y_true: torch.Tensor, profiler: BlockTimer=None):
        # inputs: 
        # links: [B, T, 17] -> (highway1, highway2, len, culm_len, start_lat, start_lon, end_lat, end_lon, POI*9)
        # dateinfo : [B, 3]
        # lens: [B]
        
        links = inputs['links_clean']
        dateinfo = inputs['dateinfo']
        lens = inputs['lens']
        
        max_len = torch.max(lens).item()
        segment_mask = torch.arange(max_len, device=lens.device).unsqueeze(0) < lens.unsqueeze(1)
        padding_mask = ~segment_mask
        
        # ENCODING THE SEGMENTS
        
        if profiler: profiler.start('enc')
        semantic_feats, len_feats, datetimerep = self.enc(links,dateinfo, profiler)  # (B, T, D), (B, T, 2), (B, datetime_dim)
        if profiler: profiler.stop()
        
        # CONTRASTIVE LEARNING
        if y_true.dim() == 2:
            y_true = y_true.squeeze(-1)
            
        if profiler: profiler.start('contrast')
        h_msm, loss_cl = self.contrast_enc(
            semantic_feats,
            src_key_padding_mask=padding_mask,
            y=y_true,
            use_contrastive=self.use_contrastive
        )
        if profiler: profiler.stop()
        
        # TEMPORAL ENCODING
        if profiler: profiler.start('GRU')
        h_modulated = self.film(h_msm, datetimerep)
        
        gru_input = torch.cat([h_modulated, len_feats], dim=-1) # (B, T, D + 2)
        gru_input = gru_input.transpose(0,1).contiguous() # (T, B, D)
        h,_ = self.temporal_block(gru_input, lens) # (B, T, D)
        h = h.transpose(0,1).contiguous()
        if profiler: profiler.stop()
        
        # ATTENTION POOLING
        if profiler: profiler.start('attn pooling')
        query = self.pool_query.expand(h.size(0), -1, -1) # (B, 1, D)
        z = self.attn(query, h, h, key_padding_mask=padding_mask)[0].squeeze(1) # (B, D)
        if profiler: profiler.stop()
        
        # REGRESSION
        z = self.pre_regression_norm(z)
        z_time = torch.concat([z, datetimerep], dim=-1) # (B,D + 33)
        t = self.regression_mlp(z_time) # (B, 1)
        
        return t, loss_cl
    
    