"""Executable examples for lesson 01.1; run from any working directory."""

import numpy as np
import torch

rng = np.random.default_rng(7)
X = rng.normal(size=(8, 4)).astype(np.float64)
W = rng.normal(size=(3, 4)).astype(np.float64)
b = rng.normal(size=(3,)).astype(np.float64)

Y_loop = np.zeros((8, 3))
for i in range(8):
    for j in range(3):
        Y_loop[i, j] = sum(X[i, k] * W[j, k] for k in range(4)) + b[j]

Y_numpy = X @ W.T + b
Y_torch = torch.from_numpy(X) @ torch.from_numpy(W).T + torch.from_numpy(b)
np.testing.assert_allclose(Y_loop, Y_numpy)
np.testing.assert_allclose(Y_numpy, Y_torch.numpy())
print("output shape:", Y_numpy.shape)

sample_centered = X - X.mean(axis=1, keepdims=True)
np.testing.assert_allclose(sample_centered.mean(axis=1), 0, atol=1e-14)

prediction = np.arange(8.0).reshape(8, 1)
target = np.arange(8.0)
print("wrong residual:", (prediction - target).shape)
print("correct residual:", (prediction[:, 0] - target).shape)

a = np.arange(6).reshape(2, 3)
print("transpose:\n", a.T)
print("reshape:\n", a.reshape(3, 2))


# Silent broadcasting changes the objective even for perfect predictions.
pred = np.array([[1.], [3.]])
labels = np.array([1., 3.])
assert np.mean((pred - labels) ** 2) == 2.0
assert np.mean((pred[:, 0] - labels) ** 2) == 0.0
source = np.arange(6.0)
view, independent = source[1:3], source[1:3].copy()
view[0] = 99
assert source[1] == 99 and independent[0] == 1
assert not np.array_equal(a.T, a.reshape(3, 2))
print("broadcast counterexample, transpose semantics, and storage sharing verified")
