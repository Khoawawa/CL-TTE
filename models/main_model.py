import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.encoder import SegmentEncoder
from models.blocks.LayerNormGRU import LayerNormBiGRU
from models.loss.contrastive_loss import SoftContrastiveLoss
from models.base.PositionalEncoding import PositionalEncodingIndex
class Cl_TTE(nn.Module):
    def __init__(self, d_model, nhead, seq_layer,temperature=0.1, sigma_percent=0.1):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        # segment encoding
        self.enc = SegmentEncoder(d_model)
        
        self.positional_encoder = PositionalEncodingIndex(d_model=d_model)
        # map
        encoder_final_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=True)
        self.mapper = nn.TransformerEncoder(encoder_layer, num_layers=seq_layer, norm=encoder_final_norm)
        # temporal modeling
        decoder_final_norm = nn.LayerNorm(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=True)
        self.anticipator = nn.TransformerDecoder(decoder_layer, num_layers=seq_layer // 2, norm=decoder_final_norm)
        
        mlp_in_dim = d_model + self.enc.datetime_dim
        
        self.regression_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim//2, mlp_in_dim//4),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim//4, 1)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
        
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
    def mean_pooling(self, h, mask):
        masked_h = h * mask.unsqueeze(-1).float()  # (B, T, D)
        sum_h = masked_h.sum(dim=1)  # (B, D)
        count = mask.sum(dim=1).unsqueeze(-1)  # (B, 1)
        mean_h = sum_h / count.clamp(min=1)  # (B, D)
        return mean_h
    def contrastive_branch(self, h, y_true):
        z = h[:, 0, :] # (B, D) CLS token representation
        z_proj = F.normalize(self.contrastive_mlp(z), dim=-1) # (B, D//4)
        with torch.amp.autocast('cuda',enabled=False):
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
        
        segment_rep, datetimerep = self.enc(links,dateinfo)  # (B, T, D)
        
        pos = self.positional_encoder(segment_rep, padding_mask=padding_mask) # (B, T, D)
        segment_rep = segment_rep + pos
        # summarize token
        B, T_plus_one, _ = segment_rep.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        segment_rep = torch.cat([cls_tokens, segment_rep], dim=1)  # (B, T+1, D)
        
        padding_mask_with_cls = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device), padding_mask], dim=1)  # (B, T+1)
        # LEARNING THE MAP OF THE TRAJECTORY
        map_memo = self.mapper(segment_rep, src_key_padding_mask=padding_mask_with_cls) # (B, T, D)

        # contrastive learning branch
        if self.training:
            l_cl = self.contrastive_branch(map_memo, y_true)
        else:
            l_cl = None
        # regression branch
        # CAUSAL TRASNFORMER DECODER ACT AS ANTICIPATOR (DRIVER)
        
        T = T_plus_one - 1
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, 
            device=links.device
        ).to(segment_rep.dtype) 

        driver_states = self.anticipator(
            tgt = segment_rep[:, 1:, :],          
            memory = map_memo,                    
            tgt_mask = causal_mask,
            tgt_key_padding_mask = padding_mask,  
            memory_key_padding_mask = padding_mask_with_cls 
        )
        
        last_step_indices = (lens - 1).clamp(min=0)  # (B,) ensure non-negative
        gather_indices = last_step_indices.view(B, 1, 1).expand(B, 1, self.d_model)  # (B, 1, D)
        
        h_final = driver_states.gather(1, index=gather_indices).squeeze(1)  # (B, D)
        
        # LINEAR REGRESSION
        t = self.regression_branch(h_final, datetimerep)  # (B, 1)
        
        return t, l_cl
    