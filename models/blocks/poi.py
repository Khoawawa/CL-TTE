import torch
import torch.nn as nn
import torch.nn.functional as F

class PoiResGatedFilMEncoder(nn.Module):
    def __init__(self, n_poi_groups, time_dim,embed_dim=32, n_layers=2):
        super().__init__()
        # the last embedding is for no poi presence
        self.poi_embed = nn.Embedding(n_poi_groups + 1, embed_dim, padding_idx=0)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "film": ResidualGatedFiLM(time_dim, embed_dim),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                ),
                "norm": nn.LayerNorm(embed_dim)
            })
            for _ in range(n_layers)
        ])
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        poi_ids = torch.arange(1,n_poi_groups + 1)
    
        self.register_buffer("poi_ids", poi_ids)
        
    
    def forward(self, poi_counts, time_embed):
        # poi_counts : (B, T, n_poi_groups)
        # time_embed: (B, T, d_model)
        B, T, P = poi_counts.shape
        poi_ids = self.poi_ids.to(poi_counts.device)
        x = self.poi_embed(poi_ids)
        x = x.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1) # (B, T, n_poi_groups, poi_dim)
        
        scale = torch.tanh(torch.log1p(poi_counts))
        x = x * (1 + scale.unsqueeze(-1))
        
        for layer in self.layers:
            x = layer["film"](x, time_embed)
            x = x + layer["ffn"](layer["norm"](x))
        
        out = x.sum(dim=2) # (B, T, poi_dim)
        
        regularized_out = self.proj(out)
        return regularized_out
        
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
    def forward(self, poi_embed, time_embed):
        gamma, beta, gate = self.proj(time_embed).chunk(3, dim=-1)
        
        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)
        gate = torch.sigmoid(gate).unsqueeze(2)
        
        poi_norm = self.norm(poi_embed)
        
        modulated = (1 + gamma) * poi_norm + beta
        
        gated = gate * modulated + (1 - gate) * poi_norm
        res_w = torch.sigmoid(self.residual_weight)
        return poi_embed + res_w * gated # (B, T, n_pois, poi_dim)

