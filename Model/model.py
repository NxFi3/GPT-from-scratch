import torch.nn.functional as F 
import torch.nn as nn
import torch
import math
from Model.utils import RMSNorm
from Model.TransformersBlock import Block



class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12, n_groups=4, d_ff=None, dropout=0.0):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, n_groups, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        
        self._tie_weights()

    def _tie_weights(self):
        """Tie the lm_head weight with the embed weight"""
        self.lm_head.weight = self.embed.weight

    def forward(self, x=None, input_ids=None,attention_mask=None, padding_mask=None, cache=None, **kwargs):
        if input_ids is not None:
            x = input_ids

        if x is None:
            raise ValueError("No input provided")

        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)

        past_caches = []

        if cache is not None:
            cache = list(cache)

        if cache is None:
            cache = [None] * len(self.blocks)

        for i, blk in enumerate(self.blocks):
            layer_cache = cache[i] if i < len(cache) else None
            x, new_cache = blk(x,attention_mask, padding_mask, layer_cache)
            past_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, tuple(past_caches)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate_manual(self, input_ids, max_new_tokens=100, temperature=0.7, 
                        do_sample=True, stop_token=None, attention_mask=None, **kwargs):
 
        with torch.no_grad():
            generated = input_ids.clone()
            cache = None
            
            for step in range(max_new_tokens):

                input_tokens = generated[:, -1:] if step > 0 else generated

                
                logits, cache = self.forward(
                    input_ids=input_tokens,
                    attention_mask=attention_mask,
                    cache=cache,
                    padding_mask=None
                )
                
                next_token_logits = logits[0, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).unsqueeze(0)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if stop_token is not None and next_token.item() == stop_token:
                    break
        
        return generated