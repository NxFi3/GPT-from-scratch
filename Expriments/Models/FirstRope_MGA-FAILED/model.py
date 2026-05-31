import torch.nn as nn
import torch.nn.functional as F 
import torch 

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):

        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
    
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
    
class RoPE(nn.Module):
    def __init__(self, head_dim, base=100):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
    def forward(self, q, k):
        """
        q, k: [B, H, S, D_head]
        returns: q_rot, k_rot (همون shape)
        """
        B, H, S, D = q.shape
        device = q.device
        pos = torch.arange(S, device=device).float()
        inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2, device=device).float() / D))
        angles = pos[:, None] * inv_freq[None, :]  # [S, D/2]

        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # [1, 1, S, D/2]
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        def rotate(x):
            x1 = x[..., 0::2]  # %2
            x2 = x[..., 1::2]  # %1
            x1_rot = x1 * cos - x2 * sin
            x2_rot = x1 * sin + x2 * cos
            return torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)
        q_rot = rotate(q)
        k_rot = rotate(k)
        return q_rot, k_rot

class MultiQueryAttention(nn.Module):
    def __init__(self, D_model, num_heads, dropout=0.1):
        super().__init__()
        assert D_model % num_heads == 0
        self.D_head = D_model // num_heads
        self.num_heads = num_heads
        self.rope = RoPE(self.D_head)
        self.q_proj = nn.Linear(D_model, D_model)
        self.k_proj = nn.Linear(D_model, self.D_head)  
        self.v_proj = nn.Linear(D_model, self.D_head) 
        self.out_proj = nn.Linear(D_model, D_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, padding=None):
        B, S, _ = x.shape
        
        Q = self.q_proj(x).view(B, S, self.num_heads, self.D_head).transpose(1, 2)
        K = self.k_proj(x).unsqueeze(1) 
        V = self.v_proj(x).unsqueeze(1) 
        Q,K = self.rope(Q,K)
   
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.D_head ** 0.5)
        
        if mask is not None:
            causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, -1e9)
        
        if padding is not None:
            padding_mask = padding.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~padding_mask, -1e9)
        
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V) 
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, D_model, num_heads, D_FF, dropout=0.1):
        super().__init__()
        self.attention = MultiQueryAttention(D_model, num_heads, dropout)
        self.FFN = SwiGLU(D_model,D_FF)
        self.norm1 = RMSNorm(D_model)
        self.norm2 = RMSNorm(D_model)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, mask=None, padding=None):
        # Pre-Norm + Attention + Residual with scale
        atten = self.attention(self.norm1(x), mask, padding)
        x = x +  self.dropout(atten)
        
        # Pre-Norm + FFN + Residual with scale
        ff = self.FFN(self.norm2(x))
        x = x + self.dropout(ff)
        return x
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size,bias=False)
        self.norm = RMSNorm(d_model)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)  
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, x, mask=None, padding=None):
        x = self.embedding(x) 
        
        for block in self.blocks:                    
            x = block(x, mask, padding)
        x = self.norm(x)                       
        return self.lm_head(x)
    
