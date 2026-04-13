import torch
import torch.nn as nn
import torch.nn.functional as F

class PoiEncoder(nn.Module):
    def __init__(self, n_poi_groups, embed_dim=32):
        super().__init__()
        # the last embedding is for no poi presence
        self.poi_embed = nn.Embedding(n_poi_groups + 1, embed_dim, padding_idx=0)
        
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        poi_ids = torch.arange(1,n_poi_groups + 1)
    
        self.register_buffer("poi_ids", poi_ids)
        
    
    def forward(self, poi_counts):
        # poi_counts : (B, T, n_poi_groups)
        # time_embed: (B, T, d_model)
        B, T, _ = poi_counts.shape
        
        poi_ids = self.poi_ids.to(poi_counts.device)
        
        x = self.poi_embed(poi_ids)
        x = x.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1) # (B, T, n_poi_groups, poi_dim)
        
        scale = torch.tanh(torch.log1p(poi_counts))
        x = x * (1 + scale.unsqueeze(-1))
        
        x = x.sum(dim=2)
        
        return self.proj(x)
class GlobalFiLM(nn.Module):
    def __init__(self, time_dim, embed_dim, n_layers):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "film": ResidualGatedFiLM(time_dim, embed_dim),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.Dropout(0.1)
                ),
                "norm": nn.LayerNorm(embed_dim)
            })
            for _ in range(n_layers)
        ])
        
        self.post_norm = nn.LayerNorm(embed_dim)
       

    def forward(self, x, time_embed):
        # x: (B, T, D)
        # time_embed: (B, T,D_time)

        for layer in self.layers:
            x = layer["film"](x, time_embed)
            x = x + layer["ffn"](layer["norm"](x))
        
        return self.post_norm(x)
        
        
class ResidualGatedFiLM(nn.Module):
    def __init__(self, time_dim, poi_dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(poi_dim, elementwise_affine=False)
        
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, poi_dim * 3)
        )
        
        self.residual_weight = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)
    def forward(self, x, time_embed):
        # x: (B, T, D)
        # time_embed: (B,D_time)
        
        gamma, beta, gate = self.proj(time_embed).chunk(3, dim=-1) # (B, D_time)
        
        gamma = gamma.unsqueeze(1) # (B, 1, D_time)
        beta = beta.unsqueeze(1) # (B, 1, D_time)
        gate = torch.sigmoid(gate).unsqueeze(1) # (B, 1, D_time)
        
        x_norm = self.norm(x)
        
        modulated = (1 + gamma) * x_norm + beta
        
        gated = gate * modulated + (1 - gate) * x_norm
        
        res_w = torch.sigmoid(self.residual_weight)
        
        return x + res_w * gated # (B, T, n_pois, poi_dim)

