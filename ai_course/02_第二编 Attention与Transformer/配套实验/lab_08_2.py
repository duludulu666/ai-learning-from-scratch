"""Runnable examples for lesson 08.2."""

import math
import torch
from torch import nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        assert width % heads == 0
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x):
        B, T, D = x.shape
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        def split(tensor):
            return tensor.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        Q, K, V = split(Q), split(K), split(V)
        weights = (Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(-1)
        read = weights @ V
        merged = read.transpose(1, 2).reshape(B, T, D)
        return self.out(merged), weights

torch.manual_seed(25)
model = MultiHeadSelfAttention(12, 3).double()
reference = nn.MultiheadAttention(12, 3, dropout=0.0, batch_first=True).double()
with torch.no_grad():
    reference.in_proj_weight.copy_(model.qkv.weight)
    reference.in_proj_bias.copy_(model.qkv.bias)
    reference.out_proj.weight.copy_(model.out.weight)
    reference.out_proj.bias.copy_(model.out.bias)

x1 = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
x2 = x1.detach().clone().requires_grad_()
actual, attention = model(x1)
expected, expected_attention = reference(x2, x2, x2,
                                        need_weights=True, average_attn_weights=False)
torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
torch.testing.assert_close(attention, expected_attention, atol=1e-10, rtol=1e-8)
actual.square().sum().backward()
expected.square().sum().backward()
torch.testing.assert_close(x1.grad, x2.grad, atol=1e-10, rtol=1e-8)
for p, q in zip(model.parameters(), reference.parameters()):
    torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-8)
assert sum(p.numel() for p in model.parameters()) == 4 * 12**2 + 4 * 12
print("output:", actual.shape, "per-head weights:", attention.shape)
print("forward, input gradients, and parameter gradients match PyTorch MultiheadAttention")
