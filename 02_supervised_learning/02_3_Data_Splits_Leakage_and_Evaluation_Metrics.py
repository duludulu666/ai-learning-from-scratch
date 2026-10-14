"""02.3 Data Splits, Leakage, and Evaluation Metrics.

From scratch in NumPy: split 48 records grouped into 12 blocks of 4.
We split by GROUP, not by record -- otherwise records from the same group
would leak between train and validation, giving an over-optimistic score.
Result: 32 train / 8 val / 8 test, with zero overlap.
"""

import numpy as np

np.random.seed(42)

# ---------- 1. Generate grouped data: 48 records, 12 groups of 4 ----------
# groups[i] = which group record i belongs to (groups 0..11, 4 records each)
groups = np.repeat(np.arange(12), 4)

# ---------- 2. Shuffle the 12 groups randomly ----------
group_order = np.random.permutation(12)

# ---------- 3. Split groups: 8 train / 2 val / 2 test ----------
# np.split cuts the shuffled group list at positions 8 and 10
train_groups, val_groups, test_groups = np.split(group_order, [8, 10])

# ---------- 4. Map group ids back to record indices ----------
# np.isin keeps every record whose group belongs to the split
train_idx = np.flatnonzero(np.isin(groups, train_groups))
val_idx = np.flatnonzero(np.isin(groups, val_groups))
test_idx = np.flatnonzero(np.isin(groups, test_groups))

# ---------- 5. Verify: no leakage between any two sets ----------
# isdisjoint = no shared index; this is the anti-leakage guarantee
assert set(train_idx).isdisjoint(val_idx)
assert set(train_idx).isdisjoint(test_idx)
assert set(test_idx).isdisjoint(val_idx)

# ---------- 6. Verify: all 48 records covered exactly once ----------
assert len(train_idx) + len(val_idx) + len(test_idx) == 48

print("train groups:", train_groups, "samples:", len(train_idx))
print("val groups:", val_groups, "samples:", len(val_idx))
print("test groups:", test_groups, "samples:", len(test_idx))
