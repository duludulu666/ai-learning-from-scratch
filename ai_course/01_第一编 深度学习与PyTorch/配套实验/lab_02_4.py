"""Executable examples for lesson 02.4; run from any working directory."""

import numpy as np
import torch

rng = np.random.default_rng(10)
x_train = np.linspace(-1, 1, 18)
x_val = np.linspace(-1, 1, 121)
signal = lambda x: np.sin(3 * x)
y_train = signal(x_train) + rng.normal(scale=0.15, size=len(x_train))
y_val = signal(x_val) + rng.normal(scale=0.15, size=len(x_val))

for degree in [1, 3, 15]:
    A = np.vander(x_train, N=degree + 1, increasing=True)
    V = np.vander(x_val, N=degree + 1, increasing=True)
    weights, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    train_mse = np.mean((A @ weights - y_train)**2)
    val_mse = np.mean((V @ weights - y_val)**2)
    print("degree/train/validation:", degree, train_mse, val_mse)

    # The same fitted polynomial is a linear model in expanded features.
    layer = torch.nn.Linear(degree + 1, 1, bias=False).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weights[None, :]))
    pytorch_prediction = layer(torch.tensor(V)).detach().numpy()[:, 0]
    np.testing.assert_allclose(pytorch_prediction, V @ weights, atol=1e-7)

# Save plots as generated experiment artifacts.
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

output = Path(__file__).resolve().parent / "figures"
output.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, degree in zip(axes, [1, 3, 15]):
    A = np.vander(x_train, N=degree + 1, increasing=True)
    V = np.vander(x_val, N=degree + 1, increasing=True)
    weights, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    ax.scatter(x_train, y_train, s=18, label="Train observations")
    ax.plot(x_val, signal(x_val), "--", color="black", label="True signal")
    ax.plot(x_val, V @ weights, label="Fit")
    ax.set(title=f"Polynomial degree {degree}", xlabel="Input", ylim=(-2, 2))
axes[0].set_ylabel("Target / prediction")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(output / "02_4_capacity.png", dpi=150)
plt.close(fig)


# Repeated-dataset decomposition on fixed design, varying training noise.
repeat_rng = np.random.default_rng(204)
design = np.vander(x_train, N=4, increasing=True)
evaluation_design = np.vander(x_val, N=4, increasing=True)
repeated_targets = signal(x_train)[:, None] + repeat_rng.normal(
    scale=0.15, size=(len(x_train), 100))
repeated_weights, *_ = np.linalg.lstsq(design, repeated_targets, rcond=None)
predictions = (evaluation_design @ repeated_weights).T
average_prediction = predictions.mean(0)
squared_bias = (average_prediction - signal(x_val)) ** 2
prediction_variance = predictions.var(0)
signal_mse = ((predictions - signal(x_val)) ** 2).mean(0)
np.testing.assert_allclose(signal_mse, squared_bias + prediction_variance, atol=1e-12)
print("empirical mean squared bias / variance / fresh noise variance:",
      squared_bias.mean(), prediction_variance.mean(), 0.15**2)
