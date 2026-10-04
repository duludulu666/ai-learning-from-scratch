"""Minimal causal decoder defined in lesson 09.4."""

import math
import torch
from torch import nn

class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x):
        B, T, D = x.shape
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        def split(z):
            return z.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        Q, K, V = split(Q), split(K), split(V)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        read = probabilities @ V
        return self.out(read.transpose(1, 2).reshape(B, T, D))

class DecoderBlock(nn.Module):
    def __init__(self, width, heads, ff_width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ff_width), nn.GELU(),
                                  nn.Linear(ff_width, width))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

class TinyTransformer(nn.Module):
    def __init__(self, vocab=9, width=32, heads=4, ff_width=64,
                 depth=2, max_length=16):
        super().__init__()
        self.max_length = max_length
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList([DecoderBlock(width, heads, ff_width)
                                     for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        if T < 1 or T > self.max_length:
            raise ValueError("input length outside configured range")
        positions = torch.arange(T, device=ids.device)
        x = self.token(ids) + self.position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, prompt, new_tokens):
        if new_tokens < 0 or prompt.shape[1] + new_tokens > self.max_length:
            raise ValueError("requested generation exceeds configured range")
        self.eval()
        result = prompt.clone()
        for _ in range(new_tokens):
            next_id = self(result)[:, -1].argmax(dim=-1, keepdim=True)
            result = torch.cat([result, next_id], dim=1)
        return result
