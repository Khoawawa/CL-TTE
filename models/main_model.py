import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.encoder import FullEncoder
from models.blocks.LayerNormGRU import LayerNormBiGRU

class Cl_TTE(nn.Module):
    def __init__(self, d_model, nhead, nlayer, seq_layer,temperature=0.1):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        # segment encoding
        self.enc = FullEncoder(d_model,nlayer=nlayer)
        # temporal modeling
        self.temporal_block = LayerNormBiGRU(input_dim=d_model, hidden_dim=d_model, num_layers=seq_layer)
        # attention pooling
        attn_dim = d_model * 2
        self.attn_norm = nn.LayerNorm(attn_dim)
        self.attention_layer = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=nhead, batch_first=True)
        
        self.sum_pooler = lambda x, mask: (x * mask.unsqueeze(-1)).sum(dim=1)
        
        mlp_in_dim = attn_dim + self.enc.datetime_dim
        
        self.regression_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim//2, mlp_in_dim//4),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim//4, 1)
        )
        
        self.contrastive_mlp = nn.Sequential(
            nn.Linear(attn_dim, attn_dim//2),
            nn.LeakyReLU(),
            nn.Linear(attn_dim//2, attn_dim//4)
        )
        
        
    def regression_branch(self, z, start_time):
        z_time = torch.concat([z, start_time], dim=-1) # (B,D + 33)
        return self.regression_mlp(z_time)
    
    def contrastive_branch(self, z_all, y_true, sigma_percent=0.1):
        """
        Fixed for Label-Relative Soft Contrastive Loss.
        sigma_percent: 0.1 means similarity drops significantly beyond a 10% time diff.
        """
        # z_all: (B, D), y_true: (B,)
        B = y_true.size(0)
        
        # 1. Project and Normalize (Standard CL practice)
        z_proj = self.contrastive_mlp(z_all) 
        z_proj = F.normalize(z_proj, p=2, dim=-1) 
        
        # 2. Compute Similarity Matrix
        y_i = y_true.unsqueeze(1)
        y_j = y_true.unsqueeze(0)

        d = torch.abs(y_i - y_j) / (0.5 * (y_i + y_j) + 1e-6)
        
        # Gaussian Kernel for Soft Weights: S_ij = exp(-|yi - yj|^2 / (2 * sigma^2))
        soft_weights = torch.exp(- (d**2) / (2 * sigma_percent**2))
        soft_weights.fill_diagonal_(0) # Exclude self-contrast
        
        # 3. Compute Similarity Matrix (Logits)
        # Dot product similarity scaled by temperature
        sim_logits = torch.matmul(z_proj, z_proj.T) / self.temperature # (B, B)
        
        # Mask diagonals for the denominator (LogSumExp)
        mask = torch.eye(B, device=y_true.device).bool()
        sim_logits_masked = sim_logits.masked_fill(mask, float('-inf'))
        
        # 4. Calculate Log-Softmax (Log-Probs)
        # log( exp(zi*zj) / sum(exp(zi*zk)) )
        log_prob = sim_logits_masked - torch.logsumexp(sim_logits_masked, dim=1, keepdim=True)
        
        # 5. Weighted Contrastive Loss
        # Instead of mean(log_prob[pos]), we take the weighted sum
        # Samples with similar y will have soft_weights close to 1
        weighted_log_prob = soft_weights * log_prob
        
        # Normalize by the sum of weights per anchor to keep gradients stable
        sum_weights = soft_weights.sum(dim=1) # (B,)
        
        valid_anchors = sum_weights > 1e-6
        if valid_anchors.any():
            loss = -(weighted_log_prob[valid_anchors].sum(dim=1) / sum_weights[valid_anchors]).mean()
        else:
            # Fallback if batch is somehow empty or identical
            loss = torch.tensor(0.0, device=y_true.device, requires_grad=True)
        
        return loss
        
        
        
    def forward(self, inputs, y_true, args):
        # inputs: 
        # links: [B, T, 7] -> (highway, len, culm_len, start_lat, start_lon, end_lat, end_lon)
        # dateinfo : [B, 3]
        # valid_mask: [B, T] 
        # lens: [B]
        links = inputs['links']
        dateinfo = inputs['dateinfo']
        lens = inputs['lens']
        max_len = torch.max(lens).item()
        segment_mask = torch.arange(max_len, device=lens.device).unsqueeze(0) < lens.unsqueeze(1)
        
        segment_rep, datetimerep = self.enc(links,dateinfo,lens)  # (B, T, D)
        
        h = segment_rep.transpose(0,1).contiguous() # (T, B, D)
        h,_ = self.temporal_block(h, lens) # (T, B, 2D)
        h = h.transpose(0,1).contiguous()
        h_norm = self.attn_norm(h)
        attn_output, _ = self.attention_layer(h_norm, h_norm, h_norm, key_padding_mask=~segment_mask)
        
        h_attended = h + attn_output
        z = self.sum_pooler(h_attended, segment_mask)
        
        t = self.regression_branch(z, datetimerep) # (B, 1)
        l_cl = self.contrastive_branch(z, y_true.squeeze(-1), args.sigma_percentile)
        
        return t, l_cl
    