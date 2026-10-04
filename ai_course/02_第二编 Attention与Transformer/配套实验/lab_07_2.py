"""Runnable examples for lesson 07.2."""

import math
import torch
from torch import nn

torch.manual_seed(22)
B, Tq, Tk, Dq, Dm, dk, dv = 2, 3, 5, 4, 6, 3, 2
query_source = torch.randn(B, Tq, Dq, dtype=torch.float64)
memory = torch.randn(B, Tk, Dm, dtype=torch.float64)
q_proj = nn.Linear(Dq, dk, bias=False).double()
k_proj = nn.Linear(Dm, dk, bias=False).double()
v_proj = nn.Linear(Dm, dv, bias=False).double()

Q, K, V = q_proj(query_source), k_proj(memory), v_proj(memory)
scores = Q @ K.transpose(-2, -1)
weights = scores.softmax(dim=-1)
output = weights @ V
assert scores.shape == (B, Tq, Tk)
assert output.shape == (B, Tq, dv)
torch.testing.assert_close(weights.sum(-1), torch.ones(B, Tq, dtype=torch.float64))

# Holding Q and K fixed keeps weights fixed, even if values change.
changed_values = V + 2.0
torch.testing.assert_close(weights @ changed_values, output + 2.0)

target = torch.randn_like(output)
loss = (output - target).square().mean()
loss.backward()
for name, layer in [("Q", q_proj), ("K", k_proj), ("V", v_proj)]:
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    print(name, "weight / gradient norm:", layer.weight.shape, layer.weight.grad.norm().item())

# Small hand-worked example: matching coordinates and value coordinates differ.
small_q = torch.tensor([1., 0.], dtype=torch.float64)
small_k = torch.tensor([[math.log(3), 1.], [0., 2.]], dtype=torch.float64)
small_v = torch.tensor([[2., 4.], [8., 0.]], dtype=torch.float64)
torch.testing.assert_close((small_k @ small_q).softmax(0) @ small_v,
                           torch.tensor([3.5, 3.], dtype=torch.float64))

# Deliberately compare a query's memory distribution with another query's.
print("first query weights:", weights.detach()[0, 0])
print("second query weights:", weights.detach()[0, 1])
