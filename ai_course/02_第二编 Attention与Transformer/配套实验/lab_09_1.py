"""Runnable examples for lesson 09.1."""

import math
import torch

def sinusoidal(length, width):
    assert width % 2 == 0
    positions = torch.arange(length, dtype=torch.float64)[:, None]
    frequency = torch.exp(-math.log(10000.0) *
                          torch.arange(0, width, 2, dtype=torch.float64) / width)
    result = torch.empty(length, width, dtype=torch.float64)
    result[:, 0::2] = torch.sin(positions * frequency)
    result[:, 1::2] = torch.cos(positions * frequency)
    return result

def rotate_pairs(x, positions):
    # x: (T,D), positions: (T,); adjacent pairs, even D.
    D = x.shape[-1]
    assert D % 2 == 0
    frequencies = 10000.0 ** (-torch.arange(0, D, 2, dtype=x.dtype) / D)
    angles = positions.to(x.dtype)[:, None] * frequencies
    a, b = x[:, 0::2], x[:, 1::2]
    result = torch.empty_like(x)
    result[:, 0::2] = a * angles.cos() - b * angles.sin()
    result[:, 1::2] = a * angles.sin() + b * angles.cos()
    return result

torch.manual_seed(27)
positions = torch.arange(6)
Q = torch.randn(6, 8, dtype=torch.float64)
K = torch.randn(6, 8, dtype=torch.float64)
Qr, Kr = rotate_pairs(Q, positions), rotate_pairs(K, positions)
torch.testing.assert_close(Qr.norm(dim=-1), Q.norm(dim=-1))
shift = 13
shifted_scores = rotate_pairs(Q, positions + shift) @ rotate_pairs(K, positions + shift).T
torch.testing.assert_close(Qr @ Kr.T, shifted_scores, atol=1e-12, rtol=1e-10)

i, j = 2, 5
relative_key = rotate_pairs(K[j:j+1], torch.tensor([j-i]))
torch.testing.assert_close((Qr[i] * Kr[j]).sum(), (Q[i] * relative_key[0]).sum())
position_table = sinusoidal(32, 16)
print("sinusoidal table:", position_table.shape)
print("norm preservation and relative-position identity verified")

# Verify the newly expanded sinusoidal shift derivation, pair by pair.
table = sinusoidal(20, 8)
delta = 3
for pair in range(4):
    angle = delta * 10000.0 ** (-2 * pair / 8)
    shift_matrix = torch.tensor([[math.cos(angle), math.sin(angle)],
                                 [-math.sin(angle), math.cos(angle)]], dtype=torch.float64)
    expected_pairs = table[:10, 2*pair:2*pair+2] @ shift_matrix.T
    torch.testing.assert_close(expected_pairs, table[delta:delta+10, 2*pair:2*pair+2])
torch.testing.assert_close(sinusoidal(2, 4)[1],
                           torch.tensor([math.sin(1), math.cos(1), math.sin(.01), math.cos(.01)],
                                        dtype=torch.float64))
print("sinusoidal shift matrix and numerical example verified")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4))
shown = ax.imshow(position_table.numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
ax.set(xlabel="Feature coordinate", ylabel="Position",
       title="Sinusoidal position features")
fig.colorbar(shown, ax=ax)
fig.tight_layout()
fig.savefig(output_dir / "09_1_positions.png", dpi=150)
plt.close(fig)
