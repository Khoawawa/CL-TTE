import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.encoder import SegmentEncoder, ContrastiveEncoder
from models.blocks.anticipator import GRUAnticipator
from models.loss.contrastive_loss import SoftContrastiveLoss
from models.base.PositionalEncoding import PositionalEncodingIndex

class Cl_TTE(nn.Module):
    def __init__(self, d_model, nhead, seq_layer,temperature=0.1, sigma_percent=0.1,n_poi_groups=9):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        assert seq_layer >= 2, "seq_layer should be at least 2 to have separate layers for mapping and anticipation"
        # segment encoding
        self.enc = SegmentEncoder(d_model=d_model, n_poi_groups=n_poi_groups)
        # contrastive learning
        self.contrast_enc = ContrastiveEncoder(d_model=d_model)
        
        self.positional_encoder = PositionalEncodingIndex(d_model=d_model)
        # map
        encoder_final_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=True)
        self.mapper = nn.TransformerEncoder(encoder_layer, num_layers=seq_layer, norm=encoder_final_norm)
        # temporal modeling
        self.anticipator = GRUAnticipator(d_model=d_model, num_layers=seq_layer // 2)
        
        mlp_in_dim = d_model + self.enc.datetime_dim
        
        self.pre_regression_norm = nn.LayerNorm(d_model)
        self.regression_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim//2, 1)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.contrastive_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model//2)
        )
        
        self.contrastive_loss = SoftContrastiveLoss(temperature=temperature, sigma_percent=sigma_percent, distance_metric='relative')
        
    def regression_branch(self, z, start_time):
        z_time = torch.concat([z, start_time], dim=-1) # (B,D + 33)
        return self.regression_mlp(z_time)

    def contrastive_branch(self, z, y_true):
        with torch.amp.autocast('cuda',enabled=False):
            z_proj = F.normalize(self.contrastive_mlp(z), dim=-1) # (B, D//2)
            loss = self.contrastive_loss(z_proj.float(), y_true.float())
        return loss
        
    def forward(self, inputs, y_true):
        # inputs: 
        # links: [B, T, 8] -> (highway1, highway2, len, culm_len, start_lat, start_lon, end_lat, end_lon)
        # dateinfo : [B, 3]
        # valid_mask: [B, T] 
        # lens: [B]
        links = inputs['links']
        dateinfo = inputs['dateinfo']
        lens = inputs['lens']
        
        max_len = torch.max(lens).item()
        segment_mask = torch.arange(max_len, device=lens.device).unsqueeze(0) < lens.unsqueeze(1)
        padding_mask = ~segment_mask
        
        # ENCODING THE SEGMENTS
        segment_rep_ori, datetimerep = self.enc(links,dateinfo)  # (B, T, D)
        
        # positional encoding
        segment_rep_ori = self.positional_encoder(segment_rep_ori, padding_mask=padding_mask) # (B, T, D)
        
        # cls token preparation
        B, _, _ = segment_rep_ori.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        segment_rep = torch.cat([cls_tokens, segment_rep_ori], dim=1)  # (B, T+1, D)
        padding_mask_with_cls = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device), padding_mask], dim=1)
        
        # LEARNING THE MAP OF THE TRAJECTORY
        map_memo = self.mapper(segment_rep, src_key_padding_mask=padding_mask_with_cls) # (B, 1 + T, D)
        
        # get the CLS token representation as the global representation of the trajectory
        z_map = map_memo[:, 0, :] # (B, D) CLS token representation
        
        # contrastive learning branch
        if self.training:
            l_cl = self.contrastive_branch(z_map, y_true)
        else:
            l_cl = None
            
        # Driver anticipation
        z_map = z_map.detach() # (B, D) CLS token representation
        h_final = self.anticipator(map_memo[:, 1:, :], z_map, lens)  # (B, D)
        
        h_final = self.pre_regression_norm(h_final)
        
        # LINEAR REGRESSION
        t = self.regression_branch(h_final, datetimerep)  # (B, 1)
        
        return t, l_cl
    