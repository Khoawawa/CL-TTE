import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.PositionalEncoding import PositionalEncodingIndex
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
        denom = denom.clamp(min=1e-6)
        
        return x_sum / denom
    def check_positive(self, y_true, r_percentile):
        # y_true: (B,) or (B,1)
        if y_true.dim() == 2:
            y_true = y_true.squeeze(-1)

        y_true = y_true.detach()

        # duplicate for 2 views
        y_all = torch.cat([y_true, y_true], dim=0)  # (2B,)

        # pairwise absolute distance
        dist = torch.abs(y_all.unsqueeze(0) - y_all.unsqueeze(1))  # (2B, 2B)

        # remove diagonal for percentile computation
        mask = ~torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
        dist_flat = dist[mask]

        # compute dynamic radius
        r = torch.quantile(dist_flat, r_percentile)

        # positive mask
        pos_mask = (dist <= r).float()

        # remove self-pairs
        pos_mask.fill_diagonal_(0)

        return pos_mask
    
    def contrastive_loss(self, z, pos_mask, temperature=0.05):
        # z: (2B, D) normalized
        sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

        # remove self similarity
        logits_mask = torch.ones_like(sim)
        logits_mask.fill_diagonal_(0)

        sim = sim - sim.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim) * logits_mask

        # denominator: all except itself
        denom = exp_sim.sum(dim=1, keepdim=True)  # (2B,1)

        # numerator: only positives
        num = (exp_sim * pos_mask).sum(dim=1)

        # avoid log(0)
        num = num.clamp(min=1e-8)

        loss = -torch.log(num / denom.squeeze(1))

        return loss.mean()
    
    def forward(self,x,x_aug,r_percentile,src_key_padding_mask=None,y_true=None):
        # x: (B, T, D)
        # x_aug: (B, T, D)
        valid_mask = ~src_key_padding_mask
        
        h = self.msm(x,src_key_padding_mask)
        h_aug = self.msm(x_aug,src_key_padding_mask)
        
        z = self.masked_mean_pooling(h,valid_mask) # (B, D)
        z_aug = self.masked_mean_pooling(h_aug,valid_mask) # (B, D)

        z = F.normalize(z, dim=-1)
        z_aug = F.normalize(z_aug, dim=-1)
        
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
        self.pos_enc = PositionalEncodingIndex(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=nlayer
        )
        self.after_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, src_key_padding_mask=None):
        # x: (B, T, D) features after segment encoder
        
        x = self.pos_enc(x, src_key_padding_mask)  # (B, T, D)
        h = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, D)
        h = self.after_norm(h)  # (B, T, D)
        
        return h
    
