# Linear regression from scratch (NumPy) with autograd gradient check.
# True model: y = 3x + 2 + Gaussian noise. Recovers w ~= 3, b ~= 2.

import numpy as np
import torch

np.random.seed(42)

# Generate synthetic data
X = np.random.uniform(0, 10, 100)
w_true = 3.0
b_true = 2.0
y_true = X * w_true + b_true + np.random.normal(0, 1, 100)

# Initialize parameters
w = 0.0
b = 0.0
lr = 0.01

# Training loop: manual gradient descent
for epochs in range(1000):
    y_predict = X * w + b
    error = 2 * (y_predict - y_true)
    dw = X * error 
    db = error
    w -= lr * dw.mean()
    b -= lr * db.mean()

    if (epochs + 1) % 100 == 0:
        print(f'epoch: {epochs+1}, w: {w:.4f}, b: {b:.4f}')

# Gradient check against autograd
X_t = torch.tensor(X)
y_t = torch.tensor(y_true)
w_t = torch.tensor(w, requires_grad=True)
b_t = torch.tensor(b, requires_grad=True)

loss = ((X_t * w_t + b_t - y_t) ** 2).mean()
loss.backward()

print('torch dw:', w_t.grad.item())
print('torch db:', b_t.grad.item())
print('numpy dw:', (X * error).mean(), ' numpy db:', error.mean())
