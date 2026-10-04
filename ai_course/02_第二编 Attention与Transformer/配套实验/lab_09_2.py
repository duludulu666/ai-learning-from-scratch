"""Runnable examples for lesson 09.2."""

import torch
from torch import nn

class PreLNBlock(nn.Module):
    def __init__(self, width=12, heads=3, ff_width=24):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ff_width), nn.GELU(),
                                  nn.Linear(ff_width, width))

    def forward(self, x):
        z = self.norm1(x)
        blocked = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
        read, _ = self.attn(z, z, z, attn_mask=blocked, need_weights=False)
        u = x + read
        return u + self.ffn(self.norm2(u))

torch.manual_seed(28)
block = PreLNBlock().double()
x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
out = block(x)
assert out.shape == x.shape
out.square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())

# FFN is position-wise: perturbing one token cannot change another FFN output.
with torch.no_grad():
    y = block.ffn(x)
    changed = x.detach().clone()
    changed[:, 3] += 4
    changed_y = block.ffn(changed)
torch.testing.assert_close(y[:, :3], changed_y[:, :3])
torch.testing.assert_close(y[:, 4:], changed_y[:, 4:])

sample = torch.randn(2, 4, 12, dtype=torch.float64)
norm = nn.LayerNorm(12, elementwise_affine=False).double()
manual = (sample - sample.mean(-1, keepdim=True)) / (
    sample.var(-1, unbiased=False, keepdim=True) + norm.eps).sqrt()
torch.testing.assert_close(norm(sample), manual)
# Constant-vector counterexample separates RMSNorm from LayerNorm.
constant = torch.full((3,), 2.0, dtype=torch.float64)
ln_constant = torch.nn.functional.layer_norm(constant, (3,), eps=1e-5)
rms_constant = constant / (constant.square().mean() + 1e-5).sqrt()
torch.testing.assert_close(ln_constant, torch.zeros_like(constant))
torch.testing.assert_close(rms_constant, torch.ones_like(constant), atol=2e-6, rtol=0)
print("block gradients, FFN locality, LayerNorm, and RMSNorm counterexample verified")
