"""Executable examples for lesson 02.1; run from any working directory."""

import numpy as np
import torch

rng = np.random.default_rng(2)
X = rng.normal(size=(128, 3))
truth = np.array([2., -1., 0.5])
y = X @ truth + 0.7 + rng.normal(scale=0.1, size=128)

w = rng.normal(size=3)
b = 0.2
error = X @ w + b - y
dw = X.T @ error / len(X)
db = error.mean()

xt, yt = torch.tensor(X), torch.tensor(y)
wt = torch.tensor(w, requires_grad=True)
bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
loss = 0.5 * (xt @ wt + bt - yt).square().mean()
loss.backward()
np.testing.assert_allclose(dw, wt.grad.numpy())
np.testing.assert_allclose(db, bt.grad.numpy())

w, b = np.zeros(3), 0.0
for _ in range(400):
    error = X @ w + b - y
    w -= 0.1 * (X.T @ error / len(X))
    b -= 0.1 * error.mean()

A = np.column_stack([X, np.ones(len(X))])
reference, *_ = np.linalg.lstsq(A, y, rcond=None)
np.testing.assert_allclose(np.r_[w, b], reference, atol=1e-8)
print("learned:", np.r_[w, b])
print("generating:", np.r_[truth, 0.7])


np.testing.assert_allclose(A.T @ (A @ reference - y), 0, atol=1e-10)
small_x, small_y = np.array([1., 2.]), np.array([3., 5.])
new_loss = 0.5 * np.mean((0.65 * small_x + 0.4 - small_y) ** 2)
np.testing.assert_allclose(new_loss, 3.673125)

# Duplicate columns identify a sum of coefficients, not each coefficient.
duplicate = np.column_stack([X[:, 0], X[:, 0], np.ones(len(X))])
solution, *_ = np.linalg.lstsq(duplicate, y, rcond=None)
alternative = solution + np.array([1., -1., 0.])
np.testing.assert_allclose(duplicate @ solution, duplicate @ alternative)
assert np.linalg.norm(solution) < np.linalg.norm(alternative)
print("normal-equation residual, hand update, and rank-deficient solutions verified")
