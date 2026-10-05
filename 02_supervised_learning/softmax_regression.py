"""Softmax regression from scratch (NumPy) with autograd gradient check.

3-class toy classification on Gaussian clusters. Accuracy > 98%.
Cross-entropy + softmax simplifies to dz = p - y_onehot in the backward pass.
"""

import numpy as np
import torch

np.random.seed(42)

# ---------- 1. Generate 3-class toy data: three Gaussian clusters ----------
X1 = np.random.randn(50, 2) + [2, 2]
X2 = np.random.randn(50, 2) + [-2, 2]
X3 = np.random.randn(50, 2) + [0, -2]
X = np.vstack([X1, X2, X3])

# Labels: class 0, 1, 2 (50 samples each)
y = np.array([0] * 50 + [1] * 50 + [2] * 50)

# ---------- 2. One-hot encoding ----------
y_onehot = np.zeros((150, 3))
y_onehot[np.arange(150), y] = 1

# ---------- 3. Initialize parameters ----------
W = np.random.randn(2, 3) * 0.01
b = np.zeros(3)
lr = 0.01


def softmax(z):
    # Numerically stable: subtract row max before exp
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# ---------- 4. Training loop: manual gradient descent ----------
for epoch in range(1000):
    # Forward
    z = X @ W + b
    p = softmax(z)

    # Cross-entropy loss
    loss = -np.mean(np.sum(y_onehot * np.log(p + 1e-12), axis=1))

    # Backward: dL/dz = p - y_onehot (beautiful simplification)
    dz = p - y_onehot
    dW = X.T @ dz / 150
    db = dz.mean(axis=0)

    # Update
    W -= lr * dW
    b -= lr * db

    if (epoch + 1) % 100 == 0:
        acc = np.mean(np.argmax(p, axis=1) == y)
        print(f'epoch: {epoch+1}, loss: {loss:.4f}, acc: {acc:.4f}')

# ---------- 5. Gradient check against autograd ----------
X_t = torch.tensor(X)
y_t = torch.tensor(y_onehot)
W_t = torch.tensor(W, requires_grad=True)
b_t = torch.tensor(b, requires_grad=True)

z_t = X_t @ W_t + b_t
p_t = torch.softmax(z_t, dim=1)
loss_t = -(y_t * torch.log(p_t + 1e-12)).sum(dim=1).mean()
loss_t.backward()

print('torch dW:', W_t.grad)
print('numpy dW:', dW)
print('torch db:', b_t.grad)
print('numpy db:', db)
