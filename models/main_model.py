import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.encoder import SegmentEncoder, ContrastiveEncoder
from models.blocks.LayerNormGRU import LayerNormGRU
import copy
class Cl_TTE(nn.Module):
    def __init__(self, d_model, nhead, nlayer, seq_layer,decoder_layer):
        super().__init__()
        self.d_model = d_model
        self.segment_encoder = SegmentEncoder(
            d_model= d_model
        )
        self.alpha_h = nn.Parameter(torch.tensor(0.2))
        self.contrastive = ContrastiveEncoder(
            d_model=d_model,
            nhead=nhead,
            nlayer=nlayer
        )
        self.post_norm = nn.LayerNorm(d_model)
        self.temporal_block = LayerNormGRU(input_dim=d_model, hidden_dim=d_model, num_layers=seq_layer)
        
        decoder_head = 1
        
        self.decoder = Decoder(d_model=d_model, N=decoder_layer, heads=decoder_head)
        
        mlp_in_dim = d_model + self.segment_encoder.datetime_dim
        self.regression_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim//2, 1)
        )
        
    def forward(self, inputs, y_true, args):
        # inputs: 
        # links: [B, T, 7] -> (highway, len, culm_len, start_lat, start_lon, end_lat, end_lon)
        # dateinfo : [B, 3]
        # valid_mask: [B, T] 
        # lens: [B]
        # y_true: [B, 1] --> logged gt
        links = inputs['links']
        dateinfo = inputs['dateinfo']
        lens = inputs['lens'].long()
        max_len = torch.max(lens).item()
        segment_mask = torch.arange(max_len, device=lens.device).unsqueeze(0) < lens.unsqueeze(1)
        
        segment_rep, datetimerep = self.segment_encoder(links,dateinfo,lens)  # (B, T, D)
        
        h, l_cl = self.contrastive(segment_rep, lens,args.mask_prob, args.data_config['noise'], args.data_config['r_percentile'], y_true) # (B, T, D)
        gate = torch.sigmoid(self.alpha_h)
        
        h = segment_rep + gate * h
        
        h = h.transpose(0,1).contiguous() # (T, B, D)
        h,_ = self.temporal_block(h, lens) # (B, T, D)
        
        with torch.amp.autocast(device_type="cuda" if h.is_cuda else "cpu", enabled=False):
            d = self.decoder(h.float(), inputs['lens'].long())
        d = d.transpose(0,1).contiguous()
        
        z =  (d * segment_mask.unsqueeze(-1)).sum(dim=1) # (B, D)
        
        z_time = torch.concat([z, datetimerep], dim=-1) # (B,D + 33)
        
        t = self.regression_mlp(z_time) # (B, 1)

        
        return t, l_cl
    


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.h = heads
        self.attn_1 = nn.MultiheadAttention(embed_dim=d_model,kdim=d_model,vdim=d_model, dropout=dropout, num_heads=self.h)

    def forward(self, q, k, v, len):
        # perform linear operation and split into N heads
        device = len.device
        max_len = torch.max(len).item()
        mask = torch.arange(max_len, device=device).unsqueeze(0) < len.unsqueeze(1)
        attn_output = self.attn_1(q, k, v, key_padding_mask=~mask, need_weights=False)[0]
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        d_ff = d_model * 2
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.ffn(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads=1, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)


    def forward(self, x, len):
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x1, x1, x1, len))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Decoder(nn.Module):
    def __init__(self, d_model, N=3, heads=1, dropout=0.1):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, lens):
        for i in range(self.N):
            x = self.layers[i](x, lens)
        return self.norm(x)