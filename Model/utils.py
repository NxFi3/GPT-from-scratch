import torch 
import torch.nn as nn
import torch.nn.functional as F



class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class RoPE(nn.Module):
    def __init__(self, head_dim, base=10000, scale=8.0):
        super().__init__()
        inv_freq = 1.0 / ((base * scale) ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k, start_pos=0):
        B, H, S, D = q.shape
        device = q.device

        pos = torch.arange(start_pos, start_pos + S, device=device).float()
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)

        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]

        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack(
                [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                dim=-1
            ).flatten(-2)

        return rotate(q), rotate(k)