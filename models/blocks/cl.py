import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.PositionalEncoding import PositionalEncodingIndex, RotaryPositionalEmbeddings

from models.blocks.moco import MoCo

class MSM(nn.Module):
    def __init__(self, d_model,nhead, dropout=0.1, nlayer=4):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dropout)
            for _ in range(nlayer)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, src_key_padding_mask=None):
        # x: (B, T, D)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)

class RoPEAttention(nn.Module):
    def __init__(self, d_model,nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        
        self.out = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)
        
        
        self.rope = RotaryPositionalEmbeddings(self.head_dim)
    
    def forward(self, x, src_key_padding_mask=None):
        # x: (B, T, D)
        # src_key_padding_mask: (B, T) with True for padding positions
        B, T, D = x.shape
        
        qkv = self.qkv(x).chunk(3, dim=-1) 
        q,k,v = [
            t.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            for t in qkv
        ] # (B, H, T, D_head)
        
        q = self.rope(q)
        k = self.rope(k)
        attn_mask = None
        if src_key_padding_mask is not None:
            attn_mask = (~src_key_padding_mask)[:, None, None, :] # (B, 1, 1, T)
            
        out = F.scaled_dot_product_attention(
            q,k,v,
            dropout_p=self.dropout if self.training else 0,
            attn_mask=attn_mask
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        return self.out_dropout(self.out(out))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = RoPEAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        x = x + self.dropout1(self.attn(self.norm1(x), src_key_padding_mask=src_key_padding_mask))
        
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        
        return x