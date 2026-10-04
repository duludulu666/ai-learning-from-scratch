"""Executable examples for lesson 02.3; run from any working directory."""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

rng = np.random.default_rng(5)
groups = np.repeat(np.arange(12), 4)  # Four related records per group
group_order = rng.permutation(12)
train_groups, val_groups, test_groups = np.split(group_order, [8, 10])
train_idx = np.flatnonzero(np.isin(groups, train_groups))
val_idx = np.flatnonzero(np.isin(groups, val_groups))
test_idx = np.flatnonzero(np.isin(groups, test_groups))

assert set(groups[train_idx]).isdisjoint(groups[val_idx])
assert set(groups[train_idx]).isdisjoint(groups[test_idx])
assert set(groups[val_idx]).isdisjoint(groups[test_idx])

X = rng.normal(size=(48, 3))
X[val_idx] += 1.0  # A deliberate validation distribution shift
mean = X[train_idx].mean(axis=0)
std = X[train_idx].std(axis=0)
normalized = (X - mean) / np.maximum(std, 1e-8)
np.testing.assert_allclose(normalized[train_idx].mean(axis=0), 0, atol=1e-14)
print("validation feature means:", normalized[val_idx].mean(axis=0))

truth = np.array([1, 1, 1, 1, 0, 0, 0, 0])
score = np.array([0.9, 0.8, 0.6, 0.2, 0.7, 0.4, 0.3, 0.1])
for threshold in [0.3, 0.5, 0.8]:
    prediction = (score >= threshold).astype(int)
    # Rows are true classes; columns are predicted classes.
    confusion = np.bincount(2 * truth + prediction, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = confusion.ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    print(threshold, confusion.tolist(), precision, recall, f1)

dataset = TensorDataset(torch.tensor(normalized[train_idx], dtype=torch.float32))
loader = DataLoader(dataset, batch_size=7)
assert sum(len(batch[0]) for batch in loader) == len(train_idx)


# The hand-worked metric example and explicit zero-denominator convention.
counts = np.array([[86, 4], [2, 8]])
tn, fp, fn, tp = counts.ravel()
np.testing.assert_allclose([(tp+tn)/counts.sum(), tp/(tp+fp), tp/(tp+fn),
                           2*tp/(2*tp+fp+fn)], [0.94, 2/3, 0.8, 8/11])
all_negative = np.zeros_like(truth)
conf = np.bincount(2*truth + all_negative, minlength=4).reshape(2, 2)
assert conf[0, 1] + conf[1, 1] == 0
assert conf[1, 1] / max(conf[:, 1].sum(), 1) == 0

# Perturb held-out values: training-fitted statistics must not change.
changed = X.copy()
changed[val_idx] += 1000
changed[test_idx] -= 1000
np.testing.assert_array_equal(changed[train_idx].mean(0), mean)
np.testing.assert_array_equal(changed[train_idx].std(0), std)
print("metric conventions and preprocessing isolation verified")
