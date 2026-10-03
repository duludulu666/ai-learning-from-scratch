# Polynomial overfitting demo from scratch (NumPy).
# True signal y = sin(3x): degree 1 underfits, 3 fits, 15 overfits.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate sparse train set (18 pts) and dense val set (121 pts)
x_train = np.linspace(-1, 1, 18)
x_val = np.linspace(-1, 1, 121)
y_train = np.sin(3 * x_train) + np.random.normal(0, 0.15, len(x_train))
y_val = np.sin(3 * x_val) + np.random.normal(0, 0.15, len(x_val))

# Build polynomial features: (n, degree+1), column j is x**j
def poly_features(x, degree):
    cols = np.zeros((len(x), degree + 1))
    for j in range(degree + 1):
        cols[:, j] = x ** j
    return cols

# Fit each degree, report MSE, and plot train / signal / fit side by side
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, degree in zip(axes, [1, 3, 15]):
    A = poly_features(x_train, degree)
    V = poly_features(x_val, degree)
    weights, *_ = np.linalg.lstsq(A, y_train, rcond=None)

    train_mse = np.mean((A @ weights - y_train) ** 2)
    val_mse = np.mean((V @ weights - y_val) ** 2)
    print("degree:", degree, "train mse:", train_mse, "val mse:", val_mse)

    ax.scatter(x_train, y_train, s=18, label="Train")
    ax.plot(x_val, np.sin(3 * x_val), "--", color="black", label="True signal")
    ax.plot(x_val, V @ weights, label="Fit")
    ax.set(title=f"degree: {degree}", ylim=(-2, 2))

axes[0].set_ylabel("y")
axes[0].legend(fontsize=7)
plt.show()
