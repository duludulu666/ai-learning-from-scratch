"""Runnable examples for lesson 08.3."""

import math
import torch
import torch.nn.functional as F

def masked_attention(Q, K, V, allowed):
    # Explicit teaching contract: Q/K/V are (B,h,T,features).
    if any(t.ndim != 4 for t in (Q, K, V)):
        raise ValueError("Q/K/V must have four named axes")
    if Q.shape[:2] != K.shape[:2] or K.shape[:3] != V.shape[:3]:
        raise ValueError("Q/K/V batch, head, or memory-length mismatch")
    if Q.shape[-1] != K.shape[-1] or K.shape[-2] == 0:
        raise ValueError("query/key widths must match; memory must be nonempty")
    if allowed.dtype != torch.bool or allowed.ndim not in (2, 4):
        raise ValueError("allowed must be Boolean (Tq,Tk) or (B,1 or h,Tq,Tk)")
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    allowed = allowed.expand_as(scores)
    has_key = allowed.any(dim=-1, keepdim=True)
    masked = scores.masked_fill(~allowed, -torch.inf)
    safe = torch.where(has_key, masked, torch.zeros_like(masked))
    weights = safe.softmax(-1)
    weights = torch.where(has_key, weights, torch.zeros_like(weights))
    return weights @ V, weights

torch.manual_seed(26)
B, h, T, d = 2, 2, 5, 3
Q = torch.randn(B, h, T, d, dtype=torch.float64)
K = torch.randn_like(Q)
V = torch.randn_like(Q)
causal = torch.ones(T, T, dtype=torch.bool).tril()
key_valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
allowed = causal[None, None] & key_valid[:, None, None, :]
output, weights = masked_attention(Q, K, V, allowed)

reference = F.scaled_dot_product_attention(Q, K, V, attn_mask=allowed, dropout_p=0.0)
torch.testing.assert_close(output, reference, atol=1e-10, rtol=1e-8)
assert weights.masked_select(~allowed.expand_as(weights)).abs().max() == 0

# Change only future keys/values. Earlier outputs must stay identical.
K2, V2 = K.clone(), V.clone()
K2[:, :, 3:] += 100
V2[:, :, 3:] -= 100
changed, _ = masked_attention(Q, K2, V2, allowed)
torch.testing.assert_close(output[:, :, :3], changed[:, :, :3])

empty = torch.zeros(T, T, dtype=torch.bool)
zero_output, zero_weights = masked_attention(Q, K, V, empty)
torch.testing.assert_close(zero_output, torch.zeros_like(zero_output))
assert torch.isfinite(zero_weights).all()
print("mask correctness, future isolation, and empty-row policy verified")


# A missing head singleton can silently target the wrong axis when B == h.
try:
    masked_attention(Q, K, V, allowed[:, 0])
except ValueError:
    pass
else:
    raise AssertionError("ambiguous three-axis mask accepted")

# Empty rows must be safe backward as well as forward. Other rows remain active.
qg = Q.clone().requires_grad_()
kg = K.clone().requires_grad_()
vg = V.clone().requires_grad_()
mixed_allowed = allowed.expand(B, h, T, T).clone()
mixed_allowed[:, :, 2] = False
mixed_output, _ = masked_attention(qg, kg, vg, mixed_allowed)
mixed_output.sum().backward()
for variable in (qg, kg, vg):
    assert torch.isfinite(variable.grad).all()
torch.testing.assert_close(qg.grad[:, :, 2], torch.zeros_like(qg.grad[:, :, 2]))
# In example 1, padded keys/values must receive no gradient through any query.
torch.testing.assert_close(kg.grad[1, :, 3:], torch.zeros_like(kg.grad[1, :, 3:]))
torch.testing.assert_close(vg.grad[1, :, 3:], torch.zeros_like(vg.grad[1, :, 3:]))

empty_inputs = [tensor.clone().requires_grad_() for tensor in (Q, K, V)]
empty_read, _ = masked_attention(*empty_inputs, empty)
empty_read.sum().backward()
for variable in empty_inputs:
    torch.testing.assert_close(variable.grad, torch.zeros_like(variable))
print("mixed/fully empty-row backward and forbidden-key gradients verified")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for n, ax in enumerate(axes):
    ax.imshow(allowed[n, 0].numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(T), yticks=range(T), xlabel="Key position", ylabel="Query position",
           title=f"Example {n}: blue = allowed")
fig.tight_layout()
fig.savefig(output_dir / "08_3_masks.png", dpi=150)
plt.close(fig)
