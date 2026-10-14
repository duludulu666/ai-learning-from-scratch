"""02.4 Underfitting, Overfitting, and Generalization.

From scratch in NumPy: true signal y = sin(3x) observed with noise.
- degree 1: underfits (too rigid, train and val error both high)
- degree 3: fits well (train and val error both low)
- degree 15: overfits (train error ~0, but val error blows up)

We deliberately use a sparse 18-point training set and a dense 121-point
validation set so the train/val MSE gap is visible at each degree.
Fitting is done by the closed-form least-squares solution (np.linalg.lstsq),
no gradient descent needed.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- 1. Generate sparse train set (18 pts) and dense val set (121 pts) ----------
# Sparse train forces the model to memorize; dense val reveals the true fit quality
x_train = np.linspace(-1, 1, 18)
x_val = np.linspace(-1, 1, 121)
y_train = np.sin(3 * x_train) + np.random.normal(0, 0.15, len(x_train))
y_val = np.sin(3 * x_val) + np.random.normal(0, 0.15, len(x_val))


# ---------- 2. Build polynomial features: (n, degree+1), column j is x**j ----------
# Row 1, x, x^2, ..., x^degree -- ordinary least squares can then solve
# weights = argmin ||A w - y||^2
def poly_features(x, degree):
    cols = np.zeros((len(x), degree + 1))
    for j in range(degree + 1):
        cols[:, j] = x ** j
    return cols


# ---------- 3. Fit each degree, report MSE, and plot side by side ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, degree in zip(axes, [1, 3, 15]):
    A = poly_features(x_train, degree)
    V = poly_features(x_val, degree)
    # Closed-form least squares: w = (A^T A)^{-1} A^T y
    weights, *_ = np.linalg.lstsq(A, y_train, rcond=None)

    train_mse = np.mean((A @ weights - y_train) ** 2)
    val_mse = np.mean((V @ weights - y_val) ** 2)
    # degree 15: train ~0 but val ~1.7 -> the overfitting signature
    print("degree:", degree, "train mse:", train_mse, "val mse:", val_mse)

    ax.scatter(x_train, y_train, s=18, label="Train")
    ax.plot(x_val, np.sin(3 * x_val), "--", color="black", label="True signal")
    ax.plot(x_val, V @ weights, label="Fit")
    ax.set(title=f"degree: {degree}", ylim=(-2, 2))

axes[0].set_ylabel("y")
axes[0].legend(fontsize=7)
plt.show()
