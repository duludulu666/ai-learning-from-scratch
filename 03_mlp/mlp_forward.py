# MLP forward/backward from scratch (NumPy) with autograd gradient check.
# Single sample: x(4,) -> ReLU hidden(5,) -> output(2,).

import numpy as np
import torch

np.random.seed(42)

# ReLU activation: negative -> 0, positive unchanged
def relu(z):
    return np.maximum(0.0, z)

# MLP forward: x -> W1@x+b1 -> relu -> W2@a+b2 -> y
def mlp_forward(x, W1, b1, W2, b2):
    h = W1 @ x + b1
    a = relu(h)
    y = W2 @ a + b2
    return y, (x, h, a, W1, W2)  # cache for backward

# MLP backward: chain rule, gradient flows from y back to W1
def mlp_backward(y, target, cache):
    x, h, a, W1, W2 = cache
    dy = y - target
    dW2 = np.outer(dy, a)
    db2 = dy.copy()
    da = W2.T @ dy
    dh = da * (h > 0)
    dW1 = np.outer(dh, x)
    db1 = dh.copy()
    return dW1, db1, dW2, db2

# Generate random single-sample data
x = np.random.normal(size=4)
W1 = np.random.normal(size=(5, 4))
b1 = np.random.normal(size=5)
W2 = np.random.normal(size=(2, 5))
b2 = np.random.normal(size=2)
target = np.random.normal(size=2)

# Forward and manual backward
y, cache = mlp_forward(x, W1, b1, W2, b2)
dW1, db1, dW2, db2 = mlp_backward(y, target, cache)
print("y:", y)
print("dW2 shape:", dW2.shape, "dW1 shape:", dW1.shape)

# Autograd gradient check: torch automatic gradients as reference
tw1 = torch.tensor(W1, requires_grad=True)
tb1 = torch.tensor(b1, requires_grad=True)
tw2 = torch.tensor(W2, requires_grad=True)
tb2 = torch.tensor(b2, requires_grad=True)
tx = torch.tensor(x)
ttarget = torch.tensor(target)

pred = tw2 @ torch.relu(tw1 @ tx + tb1) + tb2
loss = 0.5 * (pred - ttarget).square().sum()
loss.backward()

np.testing.assert_allclose(dW1, tw1.grad.numpy(), atol=1e-12)
np.testing.assert_allclose(db1, tb1.grad.numpy(), atol=1e-12)
np.testing.assert_allclose(dW2, tw2.grad.numpy(), atol=1e-12)
np.testing.assert_allclose(db2, tb2.grad.numpy(), atol=1e-12)
print("all gradients match autograd")
