import torch
import torch.nn as nn
import torch.nn.functional as F

class PoiEncoder(nn.Module):
    def __init__(self, n_poi_groups, embed_dim=32):
        super().__init__()
        # the last embedding is for no poi presence
        self.poi_embed = nn.Embedding(n_poi_groups + 1, embed_dim, padding_idx=0)
        
        self.proj = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU()
        )
        
        self.register_buffer("poi_ids", torch.arange(1,n_poi_groups + 1))
        
        self.poi_gate = nn.Sequential(
            nn.Linear(1, n_poi_groups),
            nn.ReLU(),
            nn.Linear(n_poi_groups, embed_dim),
            nn.Sigmoid()
        )
    
    def forward(self, poi_counts, culm_lens):
        # poi_counts : (B, T, G)
        # culm_lens : (B, T, 1)
        B, T, _ = poi_counts.shape
        
        
        presence = (poi_counts > 0).float() # (B, T, G)
        embeddings = self.poi_embed(self.poi_ids) # (G, D)
        
        raw_present = presence.sum(dim=-1, keepdim=True) # (B, T, 1)
        any_poi = (raw_present > 0).float() # (B, T, 1)
        n_present = raw_present.clamp(min=1) # (B, T, 1)
        
        mean_rep = torch.matmul(presence, embeddings) / n_present # (B, T, D)
        mean_rep = mean_rep * any_poi # zero out when no pois
        
        max_rep = (presence.unsqueeze(-1) * embeddings).max(dim=2).values
        max_rep = max_rep * any_poi
        
        
        combined = torch.cat([mean_rep, max_rep], dim=-1)
        
        projected = self.proj(combined)
        
        total_lens = culm_lens[:,-1:,:]
        progress = culm_lens / (total_lens + 1e-6)
        
        gate = self.poi_gate(progress) # (B, T, D)
        
        return projected * gate

class TimeScaler(nn.Module):
    def __init__(self, time_dim, embed_dim, n_layers):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "scaler": Scaler(time_dim, embed_dim),
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
        # time_embed: (B,D_time)

        for layer in self.layers:
            h = layer["scaler"](x, time_embed)
            x = h + layer["ffn"](layer["norm"](h))
        
        return self.post_norm(x)

class Scaler(nn.Module):
    def __init__(self, time_dim, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.gamma = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, d_model)
        )
        nn.init.zeros_(self.gamma[1].weight)
        nn.init.zeros_(self.gamma[1].bias)
    def forward(self, x, time_embed):
        # x: (B, T, D)
        # time_embed: (B,D_time)
        gamma = 1 + torch.tanh(self.gamma(time_embed)) # (B, D_time)
        gamma = gamma.unsqueeze(1) # (B, 1, D_time)
        
        return self.norm(x) * gamma
        
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
        # time_embed: (B,D_time)

        for layer in self.layers:
            x = layer["film"](x, time_embed)
            x = x + layer["ffn"](layer["norm"](x))
        
        return self.post_norm(x)
        
        
class ResidualGatedFiLM(nn.Module):
    def __init__(self, time_dim, poi_dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(poi_dim, elementwise_affine=False)
        
        self.proj = nn.Linear(time_dim, poi_dim * 3)
        
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
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
        
        return x + gated

