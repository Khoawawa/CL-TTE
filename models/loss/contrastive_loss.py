import torch
import torch.nn as nn
import torch.nn.functional as F

class HardContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z, positive_mask):
        z = F.normalize(z, dim=-1, eps=1e-8).float()
        
        # 2. Similarity
        sim = torch.matmul(z, z.T) / self.temperature 
        
        # 3. Stability trick
        sim = sim - sim.max(dim=1, keepdim=True)[0]

        # 4. Create boolean masks
        logits_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        pos_mask_bool = positive_mask.bool()

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
    
class SoftContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, sigma_percent=0.1, distance_metric='absolute'):
        super().__init__()
        self.temperature = temperature
        self.sigma_percent = sigma_percent
        self.distance_metric = distance_metric
        if distance_metric not in ['absolute', 'relative']:
            raise ValueError("distance metric not supported")
        
    def forward(self, z_proj, y_true):
        # z_proj: (B, D), y_true: (B,)
        B = y_true.size(0)
        
        y_i = y_true.unsqueeze(1) # (B, 1)
        y_j = y_true.unsqueeze(0) # (1, B)
        # d = |yi - yj| / ((|yi| + |yj|)/2) to get a relative difference
        if self.distance_metric == 'relative':
            # Safe relative distance (Assumes y_true > 0 strictly)
            denom = (0.5 * (y_i + y_j)) + 1e-8
            d = torch.abs(y_i - y_j) / denom
        else:
            # Absolute distance (Safe for normalized/standardized/negative y_true)
            d = torch.abs(y_i - y_j)
            
        # S_ij = exp(-d^2 / (2 * sigma^2))
        soft_weights = torch.exp(- (d**2) / (2 * self.sigma_percent**2))
        soft_weights.fill_diagonal_(0)
        
        sim_logits = torch.matmul(z_proj, z_proj.T) / self.temperature # (B, B)
        
        mask = torch.eye(B, device=y_true.device).bool()
        sim_logits_masked = sim_logits.masked_fill(mask, -1e9)
        
        log_prob = sim_logits_masked - torch.logsumexp(sim_logits_masked, dim=1, keepdim=True)
        
        weighted_log_prob = soft_weights * log_prob
        
        # Normalize by the sum of weights per anchor to keep gradients stable
        sum_weights = soft_weights.sum(dim=1)
        
        valid_anchors = sum_weights > 1e-5
        
        if valid_anchors.any():
            loss = -(weighted_log_prob[valid_anchors].sum(dim=1) / (sum_weights[valid_anchors] + 1e-8)).mean()
        else:
            loss = (z_proj * 0).sum()
        
        return loss
        