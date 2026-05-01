import torch
import torch.nn as nn

class ReconstructionLoss(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mask_ratio = 0.15
        self.mse_loss = nn.MSELoss()
    def apply_mask(self, x, pad_mask):
        # x: (B, T, D), pad_mask: (B, T) with 1 = padding, 0 = valid
        B, T, D = x.shape
        device = x.device
        
        rand = torch.rand(B, T, device=device)
        rand[pad_mask] = 1.0
        
        masked_positions = rand < self.mask_ratio
        
        mask_token = self.mask_token.expand(B, T, D)
        x_masked = torch.where(
            masked_positions.unsqueeze(-1),
            mask_token,
            x
        )
        return x_masked, masked_positions
    
    
    def forward(self, h,x_ori, masked_positions):
        
        pred = self.recon_head(h)
        
        loss = self.mse_loss(pred[masked_positions], x_ori[masked_positions].detach())
        return loss