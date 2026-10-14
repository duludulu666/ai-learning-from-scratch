"""02.1 Linear Regression and Least Squares.

From scratch in NumPy: fit y = 3x + 2 + Gaussian noise by plain gradient
descent on the mean-squared-error loss, then verify the manual gradient
against torch autograd as a sanity check.

Why MSE gradient is what it is:
    L = mean_i (w x_i + b - y_i)^2
    dL/dw = mean_i 2(w x_i + b - y_i) x_i,   dL/db = mean_i 2(w x_i + b - y_i)
The factor 2 is harmless (absorbed into the learning rate).
"""

import numpy as np
import torch

np.random.seed(42)

# ---------- 1. Generate synthetic data ----------
# 100 points uniformly spaced in x, true line y = 3x + 2, plus N(0,1) noise
X = np.random.uniform(0, 10, 100)
w_true = 3.0
b_true = 2.0
y_true = X * w_true + b_true + np.random.normal(0, 1, 100)

# ---------- 2. Initialize parameters ----------
# Start from zero; gradient descent should recover w ~= 3, b ~= 2
w = 0.0
b = 0.0
lr = 0.01

# ---------- 3. Training loop: manual gradient descent on MSE ----------
# error here already includes the factor 2 from dL/d(w x + b)
for epochs in range(1000):
    y_predict = X * w + b
    error = 2 * (y_predict - y_true)
    dw = X * error          # elementwise: gradient contribution per sample
    db = error
    w -= lr * dw.mean()     # average over the batch, then step
    b -= lr * db.mean()

    if (epochs + 1) % 100 == 0:
        print(f'epoch: {epochs+1}, w: {w:.4f}, b: {b:.4f}')

# ---------- 4. Gradient check against autograd ----------
# Feed the final parameters to torch and compare the last-step gradient.
# If our manual dw/db match autograd, the loop above is correct.
X_t = torch.tensor(X)
y_t = torch.tensor(y_true)
w_t = torch.tensor(w, requires_grad=True)
b_t = torch.tensor(b, requires_grad=True)

loss = ((X_t * w_t + b_t - y_t) ** 2).mean()
loss.backward()

print('torch dw:', w_t.grad.item())
print('torch db:', b_t.grad.item())
print('numpy dw:', (X * error).mean(), ' numpy db:', error.mean())
