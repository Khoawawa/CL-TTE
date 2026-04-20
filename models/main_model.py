import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.encoder import SegmentEncoder, ContrastiveEncoder
from models.blocks.LayerNormGRU import LayerNormGRU


from models.profiler.profiler import BlockTimer
    
class Cl_TTE(nn.Module):
    def __init__(self, d_model, nhead, seq_layer, r_percentile=0.2,n_poi_groups=9,contrastive_temperature=0.25):
        super().__init__()
        
        self.d_model = d_model
        assert seq_layer >= 2, "seq_layer should be at least 2 to have separate layers for mapping and anticipation"
        
        # SEGMENT ENCODER
        self.enc = SegmentEncoder(d_model=d_model, n_poi_groups=n_poi_groups, nlayers=seq_layer)
        
        # CONTRASTIVE BLOCK
        self.contrast_enc = ContrastiveEncoder(d_model=d_model,nlayer=seq_layer,nhead=nhead,r_percentile=r_percentile,contrastive_temperature=contrastive_temperature)
        self.alpha_h = nn.Parameter(torch.tensor(0.2))
        
        # TEMPORAL BLOCK
        self.post_norm = nn.LayerNorm(d_model)
        self.temporal_block = LayerNormGRU(input_dim=d_model, hidden_dim=d_model, num_layers=seq_layer)
        
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
        
        links_clean = inputs['links_clean']
        links_aug = inputs['links_aug']
        src_key_aug_padding_mask = inputs['augment_mask']
        dateinfo = inputs['dateinfo']
        lens = inputs['lens']
        
        max_len = torch.max(lens).item()
        segment_mask = torch.arange(max_len, device=lens.device).unsqueeze(0) < lens.unsqueeze(1)
        padding_mask = ~segment_mask
        
        # ENCODING THE SEGMENTS
        
        links = torch.cat([links_clean, links_aug], dim=0) # (2B, T, input_dim)
        if profiler: profiler.start('enc')
        segment_rep, datetimerep = self.enc(links,dateinfo, profiler)  # (2B, T, D)
        if profiler: profiler.stop()
        
        segment_rep_clean, segment_rep_aug = torch.chunk(segment_rep, 2, dim=0) # (B, T, D)
        
        # CONTRASTIVE LEARNING
        if profiler: profiler.start('contrast')
        with torch.amp.autocast('cuda',enabled=False):
            h_cl, l_cl = self.contrast_enc(
                segment_rep_clean, 
                segment_rep_aug, 
                src_key_padding_mask=padding_mask, 
                src_key_augment_padding_mask=src_key_aug_padding_mask,
                y_true=y_true
            )
        if profiler: profiler.stop()
        
        # GATED RESIDUAL CONNECTION
        gate = torch.sigmoid(self.alpha_h)
        h = self.post_norm(segment_rep_clean + gate * h_cl.detach())
        
        # TEMPORAL ENCODING
        if profiler: profiler.start('GRU')
        h = h.transpose(0,1).contiguous() # (T, B, D)
        h,_ = self.temporal_block(h, lens) # (B, T, D)
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
        
        return t, l_cl
    