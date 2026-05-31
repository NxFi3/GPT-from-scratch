import torch 
import math 
import torch.nn as nn
import torch.nn.functional as F
from Model.utils import RoPE,RMSNorm , SwiGLU

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_groups, dropout=0.0):
        super().__init__()

        assert d_model % n_heads == 0
        assert n_heads % n_groups == 0

        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.group_size = n_heads // n_groups

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_head * n_groups, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_head * n_groups, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE(self.d_head)
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)

        self.dropout = dropout

    def forward(self, x,attention_mask=None, padding_mask=None, cache=None):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_groups, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_groups, self.d_head).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)



        seq_len = 0 
        if cache is not None:
            k_past, v_past = cache
            seq_len = k_past.size(2)
            q, k = self.rope(q, k,start_pos=seq_len)
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        else:
            q, k = self.rope(q, k,start_pos=seq_len)


        past_cache = (k,v)

        




        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True if cache is None else False,
            dropout_p=self.dropout if self.training else 0.0,
            attn_mask=None #DONT USE.
        )


        if padding_mask is not None:
            out = out.transpose(1, 2)  # [B, S, H, D]
            mask = padding_mask[:, :, None, None].to(torch.bool)  # [B, S, 1, 1]
            out = out.masked_fill(~mask, 0.0)
            out = out.transpose(1, 2)  # [B, H, S, D]

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out), past_cache


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_groups, dropout=0.0):
        super().__init__()

        self.attn = GroupedQueryAttention(d_model, n_heads, n_groups, dropout)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.res_scale = 1.0 / math.sqrt(2)

    def forward(self, x, attention_mask= None,padding_mask=None, cache=None):
        attn_out, past_cache = self.attn(self.norm1(x),attention_mask, padding_mask, cache)
        x = x + self.res_scale * attn_out
        x = x + self.res_scale * self.ffn(self.norm2(x))
        return x, past_cache