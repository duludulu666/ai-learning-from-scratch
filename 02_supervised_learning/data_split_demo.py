# Group-wise data split from scratch (NumPy).
# 12 groups x 4 records: train/val/test 32/8/8 samples, no data leakage.

import numpy as np

np.random.seed(42)

# Generate grouped data: 48 records, each belongs to one of 12 groups
groups = np.repeat(np.arange(12), 4)

# Shuffle the 12 groups randomly
group_order = np.random.permutation(12)

# Split groups: first 8 for train, next 2 for val, last 2 for test
train_groups, val_groups, test_groups = np.split(group_order, [8, 10])

# Map group ids to record indices
train_idx = np.flatnonzero(np.isin(groups, train_groups))
val_idx = np.flatnonzero(np.isin(groups, val_groups))
test_idx = np.flatnonzero(np.isin(groups, test_groups))

# Verify: any two sets are disjoint (no leakage)
assert set(train_idx).isdisjoint(val_idx)
assert set(train_idx).isdisjoint(test_idx)
assert set(test_idx).isdisjoint(val_idx)

# Verify: all 48 records covered exactly once
assert len(train_idx) + len(val_idx) + len(test_idx) == 48

print("train groups:", train_groups, "samples:", len(train_idx))
print("val groups:", val_groups, "samples:", len(val_idx))
print("test groups:", test_groups, "samples:", len(test_idx))
