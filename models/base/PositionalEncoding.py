import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import math
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, seq_len, device):
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return

        theta = 1. / (self.base ** (torch.arange(0, self.d, 2, device=device).float() / self.d))

        pos = torch.arange(seq_len, device=device).float()

        idx_theta = torch.einsum('t,d->td', pos, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)

        # (T, D)
        self.cos_cached = idx_theta2.cos().float()
        self.sin_cached = idx_theta2.sin().float()

    def _neg_half(self, x):
        d_2 = self.d // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, x):
        # x: (B, H, T, D)
        B, H, T, D = x.shape

        self._build_cache(T, x.device)

        cos = self.cos_cached[:T]  # (T, D)
        sin = self.sin_cached[:T]

        # reshape for broadcast
        cos = cos[None, None, :, :]  # (1,1,T,D)
        sin = sin[None, None, :, :]

        x_rot = self._neg_half(x)

        return x * cos + x_rot * sin
    
class CyclicalTimeEncoding(nn.Module):
    def __init__(self, d_model: int = 256, period: float = 1440.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        self.period = period
        half_dim = d_model // 2

        frequencies = torch.exp(
            torch.linspace(0, math.log(half_dim), half_dim)
        )
        self.register_buffer("frequencies", frequencies)  # (half_dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,) or (B, 1)   global trip time
        return: (B, d_model)
        """
        if x.dim() == 2:
            x = x.squeeze(1)   # (B,)
        elif x.dim() != 1:
            raise ValueError("x must have shape (B,) or (B, 1)")

        # (B, 1)
        x = x.unsqueeze(-1)

        # (B, half_dim)
        x_arg = (2 * math.pi * x.unsqueeze(-1)) / self.period * self.frequencies

        pe = torch.cat(
            [torch.sin(x_arg), torch.cos(x_arg)],
            dim=-1
        )  # (B, d_model)

        return pe

class PositionalEncodingIndex(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)          # (T, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)  # (T, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Args:
            x:              (B, T, D)
            padding_mask:   (B, T)  boolean
                            True  = **pad**   (to be masked)
                            False = valid
                            (matches the format expected by TransformerEncoderLayer src_key_padding_mask)
        """
        seq_len = x.shape[1]
        pe = self.pe[:, :seq_len, :].clone()           # (1, T, D)

        if padding_mask is not None:
            # Zero positional encoding where we will mask attention anyway
            # ~padding_mask == valid positions
            pe = pe * (~padding_mask).unsqueeze(-1).to(pe.dtype)

            # Alternative (equivalent but sometimes clearer):
            # pe = pe.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return self.dropout(x + pe)
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        half_dim = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_dim).float()
            * (-math.log(10000.0) / half_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,) or (B,1)
        return: (B, d_model)
        """

        if x.dim() == 1:   # (B,)
            x = x.unsqueeze(-1)

        # x is now (B,1)

        x_arg = x * self.div_term  # (B, d_model/2)

        pe = torch.cat(
            [torch.sin(x_arg), torch.cos(x_arg)],
            dim=-1
        )  # (B, d_model)

        return pe
    
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int = 256):
        """
        Args:
            d_model (int): The total dimensionality of the output encoding.
                           Must be divisible by 4.
        """
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 4.")
        
        self.d_model = d_model
        
        # We will split the d_model into 4 parts:
        # d_model / 4 for sin(dx), d_model / 4 for cos(dx)
        # d_model / 4 for sin(dy), d_model / 4 for cos(dy)
        self.d_half = d_model // 2
        self.d_quarter = d_model // 4
        
        # Create the 'div_term' for the denominator: 10000^(2i / (d_model/2))
        # We use d_model/2 because we apply it to dx and dy independently.
        div_term = torch.exp(torch.arange(0, self.d_half, 2).float() * \
                             (-math.log(10000.0) / (self.d_half)))
        
        # Register as a buffer so it's part of the module's state,
        # but not a trainable parameter.
        self.register_buffer('div_term', div_term)

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            offsets (torch.Tensor): A tensor of shape (batch_size, 2)
                                    containing the (dx, dy) pairs.

        Returns:
            torch.Tensor: The positional encoding of shape (batch_size, d_model).
        """
        
        # (batch_size, 1) * (d_quarter) -> (batch_size, d_quarter)
        dx_arg = offsets[:, 0:1] * self.div_term
        dy_arg = offsets[:, 1:2] * self.div_term
        
        # Apply sin/cos to dx and dy arguments
        # Each is (batch_size, d_quarter)
        pe_dx_sin = torch.sin(dx_arg)
        pe_dx_cos = torch.cos(dx_arg)
        pe_dy_sin = torch.sin(dy_arg)
        pe_dy_cos = torch.cos(dy_arg)

        # Concatenate all four parts
        # (batch_size, d_quarter * 4) -> (batch_size, d_model)
        pe = torch.cat([pe_dx_sin, pe_dx_cos, pe_dy_sin, pe_dy_cos], dim=1)
        
        return pe