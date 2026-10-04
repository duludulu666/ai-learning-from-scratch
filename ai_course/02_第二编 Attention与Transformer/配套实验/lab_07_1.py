"""Runnable examples for lesson 07.1."""

import numpy as np
import torch

keys = np.eye(3, dtype=np.float64)
values = np.array([[2., 20.], [8., 80.], [-3., -30.]])
queries = np.eye(3, dtype=np.float64)
scores = 8.0 * queries @ keys.T
shifted = scores - scores.max(axis=-1, keepdims=True)
weights = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
outputs = weights @ values
reference = torch.softmax(torch.tensor(scores), dim=-1) @ torch.tensor(values)
np.testing.assert_allclose(outputs, reference.numpy())
np.testing.assert_allclose(weights.sum(-1), 1)
assert np.max(np.abs(outputs - values)) < 0.1

print("fixed mean:", values.mean(axis=0))
print("question-dependent reads:\n", outputs)
print("weights:\n", weights)

# Reorder paired keys and values: retrieval meaning is unchanged.
permutation = [2, 0, 1]
reordered_scores = 8.0 * queries @ keys[permutation].T
reordered_weights = torch.softmax(torch.tensor(reordered_scores), dim=-1)
reordered_output = reordered_weights @ torch.tensor(values[permutation])
torch.testing.assert_close(reordered_output, reference)

for temperature in [0.2, 1.0, 5.0]:
    probabilities = torch.softmax(torch.tensor([2., 0., -1.]) / temperature, dim=-1)
    print("temperature / weights:", temperature, probabilities)

# Worked two-value examples in the lesson.
small_values = torch.tensor([2.0, 8.0], dtype=torch.float64)
small_scores = torch.tensor([np.log(3), 0.0], dtype=torch.float64)
torch.testing.assert_close(small_scores.softmax(0) @ small_values, torch.tensor(3.5, dtype=torch.float64))
torch.testing.assert_close(small_scores.flip(0).softmax(0) @ small_values, torch.tensor(6.5, dtype=torch.float64))

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(weights, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{weights[i,j]:.3f}", ha="center", va="center",
                color="white" if weights[i,j] > 0.5 else "black")
ax.set(xticks=range(3), yticks=range(3), xlabel="Memory key", ylabel="Query",
       title="Question-dependent retrieval weights")
fig.tight_layout()
fig.savefig(output_dir / "07_1_retrieval.png", dpi=150)
plt.close(fig)
