import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.PositionalEncoding import PositionalEncodingIndex, RotaryPositionalEmbeddings

# Assuming MoCo is imported or used elsewhere
from models.blocks.moco import MoCo
class MSM(nn.Module):
    def __init__(self, d_model, nhead, dropout1=0.1, dropout2=0.3, nlayer=4):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dropout1, dropout2)
            for _ in range(nlayer)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, src_key_padding_mask=None, use_heavy_dropout=False):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask, use_heavy_dropout=use_heavy_dropout)
        return self.norm(x)

class RoPEAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout1=0.1, dropout2=0.3):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        
        self.rope = RotaryPositionalEmbeddings(self.head_dim)
    
    def forward(self, x, src_key_padding_mask=None, use_heavy_dropout=False):
        B, T, D = x.shape
        
        current_p = self.dropout2 if use_heavy_dropout else self.dropout1
        
        qkv = self.qkv(x).chunk(3, dim=-1) 
        q, k, v = [
            t.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            for t in qkv
        ] 
        
        q = self.rope(q)
        k = self.rope(k)
        
        attn_mask = None
        if src_key_padding_mask is not None:
            attn_mask = (~src_key_padding_mask)[:, None, None, :] 
            
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=current_p if self.training else 0.0,
            attn_mask=attn_mask
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)
        
        return out
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout1=0.1, dropout2=0.3):
        super().__init__()
        self.attn = RoPEAttention(d_model, nhead, dropout1, dropout2)
        
        self.ffn_linear1 = nn.Linear(d_model, d_model * 4)
        self.ffn_act = nn.GELU()
        self.ffn_linear2 = nn.Linear(d_model * 4, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = dropout1
        self.dropout2 = dropout2

    def forward(self, x, src_key_padding_mask=None, use_heavy_dropout=False):
        current_p = self.dropout2 if use_heavy_dropout else self.dropout1
        
        attn_out = self.attn(self.norm1(x), src_key_padding_mask=src_key_padding_mask, use_heavy_dropout=use_heavy_dropout)
        x = x + F.dropout(attn_out, p=current_p, training=self.training)
        
        ffn_out = self.ffn_act(self.ffn_linear1(self.norm2(x)))
        ffn_out = F.dropout(ffn_out, p=current_p, training=self.training)
        
        x = x + F.dropout(self.ffn_linear2(ffn_out), p=current_p, training=self.training)
        return x