"""Executable examples for lesson 03.2; run from any working directory."""

import torch

x = torch.tensor([1., 2.], dtype=torch.float64)
target = torch.tensor([2.], dtype=torch.float64)
W1 = torch.tensor([[1., -1.], [2., 1.]], dtype=torch.float64, requires_grad=True)
b1 = torch.tensor([0., -1.], dtype=torch.float64, requires_grad=True)
W2 = torch.tensor([[2., -1.]], dtype=torch.float64, requires_grad=True)
b2 = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
h = W1 @ x + b1
a = torch.relu(h)
y = W2 @ a + b2
loss = 0.5 * (y - target).square().sum()
loss.backward()
torch.testing.assert_close(W1.grad, torch.tensor([[0., 0.], [4., 8.]], dtype=torch.float64))
torch.testing.assert_close(b1.grad, torch.tensor([0., 4.], dtype=torch.float64))
torch.testing.assert_close(W2.grad, torch.tensor([[0., -12.]], dtype=torch.float64))
torch.testing.assert_close(b2.grad, torch.tensor([-4.], dtype=torch.float64))
print("prediction/loss:", y.detach(), loss.item())


# A batched hand-written backward with all parameters and input gradients.
import numpy as np
rng = np.random.default_rng(302)
X = rng.normal(size=(3, 4))
W1 = rng.normal(size=(5, 4))
b1 = rng.normal(size=5)
W2 = rng.normal(size=(2, 5))
b2 = rng.normal(size=2)
target = rng.normal(size=(3, 2))
H = X @ W1.T + b1
A = np.maximum(H, 0)
Y = A @ W2.T + b2
GY = (Y-target) / len(X)
GH = (GY @ W2) * (H > 0)
manual = [GH.T @ X, GH.sum(0), GY.T @ A, GY.sum(0), GH @ W1]
tensors = [torch.tensor(z, requires_grad=True) for z in (W1,b1,W2,b2,X)]
w1t,b1t,w2t,b2t,xt = tensors
output = torch.relu(xt @ w1t.T+b1t) @ w2t.T+b2t
(0.5*(output-torch.tensor(target)).square().sum(-1).mean()).backward()
for calculated, variable in zip(manual, tensors):
    np.testing.assert_allclose(calculated, variable.grad.numpy(), atol=1e-11)
per_example = [np.zeros_like(z) for z in (W1,b1,W2,b2)]
for n in range(len(X)):
    gy = Y[n]-target[n]
    gh = (W2.T @ gy) * (H[n] > 0)
    contributions = [np.outer(gh, X[n]), gh, np.outer(gy,A[n]), gy]
    for accumulator, contribution in zip(per_example, contributions):
        accumulator += contribution / len(X)
for batch_gradient, averaged in zip(manual[:4], per_example):
    np.testing.assert_allclose(batch_gradient, averaged, atol=1e-12)
print("batched parameter/input gradients and per-example averaging verified")
