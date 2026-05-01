import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.PositionalEncoding import PositionalEncodingIndex, RotaryPositionalEmbeddings

from models.blocks.moco import MoCo

class ReCo(nn.Module):
    def __init__(self,d_model,nhead,dropout=0.1,nlayer=4):
        super().__init__()
        self.msm = MSM(d_model,nhead,dropout,nlayer)
        
    def masked_mean_pooling(self, x, mask):
        # x: (B, T, D)
        # mask: (B, T) with 1 = valid, 0 = padding
        mask = mask.float()
        # sum
        x_sum = torch.einsum('btd,bt->bd', x, mask)
        # count valid tokens
        denom = mask.sum(dim=1, keepdim=True)  # (B, 1)
        # avoid division by zero
        denom = denom.clamp(min=1e-4)
        return x_sum / denom
        
    def check_positive(self, y_true, r_percentile=0.2):
        if y_true.dim() == 2:
            y_true = y_true.squeeze(-1)
        y_true = y_true.detach()
        B = y_true.size(0)

        dist_orig = torch.abs(y_true.unsqueeze(0) - y_true.unsqueeze(1))
        mask = ~torch.eye(B, dtype=torch.bool, device=y_true.device)
        dist_flat = dist_orig[mask]
        
        r = torch.quantile(dist_flat, r_percentile)
        r = torch.clamp(r, min=1e-3)

        y_all = torch.cat([y_true, y_true], dim=0) 
        dist = torch.abs(y_all.unsqueeze(0) - y_all.unsqueeze(1))
        
        pos_mask = (dist <= r).float()
        pos_mask.fill_diagonal_(0)

        num_pos = pos_mask.sum(dim=1)
        if (num_pos == 0).any():
            dist_no_self = dist + torch.eye(dist.size(0), device=dist.device) * 1e4
            nn_idx = dist_no_self.argmin(dim=1)
            pos_mask[torch.arange(dist.size(0)), nn_idx] = 1.0

        return pos_mask

    def contrastive_loss(self, z, pos_mask, temperature=0.2):
        # 1. Normalize safely and force FP32
        z = F.normalize(z, dim=-1, eps=1e-8).float()
        
        # 2. Similarity
        sim = torch.matmul(z, z.T) / temperature 
        
        # 3. Stability trick
        sim = sim - sim.max(dim=1, keepdim=True)[0]

        # 4. Create boolean masks
        logits_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        pos_mask_bool = pos_mask.bool()

        # 5. Fast PyTorch Masking (-1e9 completely removes them from logsumexp)
        # Denominator: All pairs EXCEPT self
        sim_denom = sim.masked_fill(~logits_mask, -1e4)
        log_denom = torch.logsumexp(sim_denom, dim=1)

        # Numerator: ONLY positive pairs
        sim_num = sim.masked_fill(~pos_mask_bool, -1e4)
        log_num = torch.logsumexp(sim_num, dim=1)

        # 6. Loss
        loss = -(log_num - log_denom)

        # ---- FINAL SAFETY ----
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        return loss.mean()
    
    def forward(self,x,x_aug,r_percentile,src_key_padding_mask=None,y_true=None):
        # x: (B, T, D)
        # x_aug: (B, T, D)
        valid_mask = ~src_key_padding_mask
        
        h = self.msm(x,src_key_padding_mask)
        h_aug = self.msm(x_aug,src_key_padding_mask)
        
        z = self.masked_mean_pooling(h,valid_mask) # (B, D)
        z_aug = self.masked_mean_pooling(h_aug,valid_mask) # (B, D)
        
        # F.normalize is now handled safely inside contrastive_loss!
        z_all = torch.cat([z,z_aug],dim=0) # (2B, D)
        
        if y_true is None:
            l_cl = None
        else:
            pos_mask = self.check_positive(y_true,r_percentile)
            l_cl = self.contrastive_loss(z_all,pos_mask)
         
        return h, l_cl
    
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