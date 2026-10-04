"""Executable examples for lesson 01.2; run from any working directory."""

import numpy as np
import torch
from torch import nn

X = np.array([[1., 2.], [3., -1.]], dtype=np.float64)
W = np.array([[3., -1.], [2., 4.], [-1., 0.]], dtype=np.float64)
b = np.array([0.5, 0., 1.], dtype=np.float64)
Y = X @ W.T + b

layer = nn.Linear(2, 3).double()
with torch.no_grad():
    layer.weight.copy_(torch.from_numpy(W))
    layer.bias.copy_(torch.from_numpy(b))
np.testing.assert_allclose(layer(torch.from_numpy(X)).detach().numpy(), Y)

x = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
upstream = torch.tensor([1., -2., 0.5], dtype=torch.float64)
loss = (layer(x) * upstream).sum()
loss.backward()

torch.testing.assert_close(x.grad, layer.weight.T @ upstream)
torch.testing.assert_close(layer.weight.grad, torch.outer(upstream, x.detach()))
print("outputs:\n", Y)
print("input gradient:", x.grad)
print("weight gradient shape:", layer.weight.grad.shape)

J = torch.autograd.functional.jacobian(layer, x.detach())
torch.testing.assert_close(J, layer.weight)


torch.testing.assert_close(layer.bias.grad, upstream)
small_w = torch.tensor([[2., -1.], [3., 4.]], dtype=torch.float64)
small_g = torch.tensor([5., -2.], dtype=torch.float64)
torch.testing.assert_close(small_w.T @ small_g, torch.tensor([4., -13.], dtype=torch.float64))
# Composing two affine maps changes coefficients, not the function class.
rng = np.random.default_rng(102)
A, C = rng.normal(size=(4, 2)), rng.normal(size=(3, 4))
a_bias, c_bias = rng.normal(size=4), rng.normal(size=3)
np.testing.assert_allclose((X @ A.T + a_bias) @ C.T + c_bias,
                           X @ (C @ A).T + (C @ a_bias + c_bias))
print("bias gradient, hand-worked Jacobian, and affine composition verified")
