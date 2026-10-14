"""03.1 MLP and Backpropagation.

From scratch in NumPy: single-sample MLP
    x(4) -> linear(4,5) -> ReLU -> linear(5,2) -> y(2)
with hand-derived chain-rule gradients, verified against torch autograd to 1e-12.

Dimensions:
    x: (4,)   W1: (5,4)   b1: (5,)
    h: (5,)   a = relu(h): (5,)
    W2: (2,5) b2: (2,)
    y: (2,)
"""

import numpy as np
import torch

np.random.seed(42)


# ---------- 1. ReLU activation: negative -> 0, positive unchanged ----------
def relu(z):
    return np.maximum(0.0, z)


# ---------- 2. MLP forward: x -> W1@x+b1 -> relu -> W2@a+b2 -> y ----------
def mlp_forward(x, W1, b1, W2, b2):
    h = W1 @ x + b1          # pre-activation, (5,)
    a = relu(h)              # post-activation, (5,)
    y = W2 @ a + b2          # output, (2,)
    # cache everything needed for the backward pass
    return y, (x, h, a, W1, W2)


# ---------- 3. MLP backward: chain rule, gradient flows from y back to W1 ----------
def mlp_backward(y, target, cache):
    x, h, a, W1, W2 = cache
    # MSE loss L = 0.5 * ||y - target||^2, so dy = y - target
    dy = y - target                       # (2,)
    dW2 = np.outer(dy, a)                 # outer product: (2,5)
    db2 = dy.copy()                       # (2,)
    da = W2.T @ dy                        # (5,)
    # ReLU derivative: pass gradient only where h > 0 (zero where h <= 0)
    dh = da * (h > 0)
    dW1 = np.outer(dh, x)                 # outer product: (5,4)
    db1 = dh.copy()                       # (5,)
    return dW1, db1, dW2, db2


# ---------- 4. Generate random single-sample data ----------
x = np.random.normal(size=4)
W1 = np.random.normal(size=(5, 4))
b1 = np.random.normal(size=5)
W2 = np.random.normal(size=(2, 5))
b2 = np.random.normal(size=2)
target = np.random.normal(size=2)

# ---------- 5. Forward and manual backward ----------
y, cache = mlp_forward(x, W1, b1, W2, b2)
dW1, db1, dW2, db2 = mlp_backward(y, target, cache)
print("y:", y)
print("dW2 shape:", dW2.shape, "dW1 shape:", dW1.shape)

# ---------- 6. Autograd gradient check: torch as reference ----------
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
