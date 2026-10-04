"""Executable examples for lesson 06.1; run from any working directory."""

import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(2)
sequences = torch.tensor([[[0.], [1.]], [[1.], [0.]]])
torch.testing.assert_close(sequences[0].mean(0), sequences[1].mean(0))
ordered_score = sequences[:, -1, 0] - sequences[:, 0, 0]
print("same mean, different order score:", ordered_score)

# Token 0 is padding; tokens 1–4 are actual observations.
tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 0, 0]])
lengths = torch.tensor([4, 2])
inputs, targets = tokens[:, :-1], tokens[:, 1:]
valid = torch.arange(targets.shape[1])[None, :] < (lengths - 1)[:, None]
assert valid.sum().item() == 4

embedding = nn.Embedding(5, 4, padding_idx=0)
head = nn.Linear(4, 5)
logits = head(embedding(inputs))  # A position-wise baseline, no context mixing
per_token = F.cross_entropy(logits.reshape(-1, 5),
                            targets.reshape(-1), reduction="none").reshape_as(targets)
masked_loss = (per_token * valid).sum() / valid.sum()
indexed_loss = F.cross_entropy(logits[valid], targets[valid])
torch.testing.assert_close(masked_loss, indexed_loss)
masked_loss.backward()
torch.testing.assert_close(embedding.weight.grad[0], torch.zeros(4))

print("input/target/logit shapes:", inputs.shape, targets.shape, logits.shape)
print("valid targets:", targets[valid])
print("masked loss:", masked_loss.item())


def valid_cross_entropy(scores, labels, valid_mask):
    if scores.shape[:-1] != labels.shape or valid_mask.shape != labels.shape:
        raise ValueError("scores, targets, and mask shapes disagree")
    if valid_mask.dtype != torch.bool or not valid_mask.any():
        raise ValueError("at least one valid Boolean target position is required")
    return F.cross_entropy(scores[valid_mask],labels[valid_mask])

baseline_loss = valid_cross_entropy(logits.detach(),targets,valid)
altered = logits.detach().clone()
altered[~valid] = torch.nan  # Never passed to CE under indexed selection.
torch.testing.assert_close(valid_cross_entropy(altered,targets,valid),baseline_loss)
extended_scores = torch.cat([logits.detach(),torch.zeros(2,2,5)],1)
extended_targets = torch.cat([targets,torch.full((2,2),-999,dtype=torch.long)],1)
extended_mask = torch.cat([valid,torch.zeros(2,2,dtype=torch.bool)],1)
torch.testing.assert_close(valid_cross_entropy(extended_scores,extended_targets,extended_mask),
                           baseline_loss)
try:
    valid_cross_entropy(logits.detach(),targets,torch.zeros_like(valid))
except ValueError:
    pass
else:
    raise AssertionError("all-invalid target set accepted")
print("indexed loss padding invariance, invalid ignored labels, and empty rejection verified")
