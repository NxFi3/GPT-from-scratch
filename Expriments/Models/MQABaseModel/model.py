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

class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size=50256, emb_dim=128, maxlen=2048, dropout=0.01):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, emb_dim)
        self.pos = nn.Parameter(torch.randn(1, maxlen, emb_dim) * 0.01)  # ← std=0.01
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T = x.shape
        tok = self.tok(x)
        pos = self.pos[:, :T]
        return self.drop(tok + pos)
    

class MultiQueryAttention(nn.Module):
    def __init__(self, D_model, num_heads, dropout=0.1):
        super().__init__()
        assert D_model % num_heads == 0
        self.D_head = D_model // num_heads
        self.num_heads = num_heads
        
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
    def __init__(self, vocab_size, max_len, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = PositionalEmbedding(vocab_size, d_model, max_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size,bias=False)
        self.lm_head.weight = self.embedding.tok.weight
        self.norm = RMSNorm(d_model)
        
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
    
