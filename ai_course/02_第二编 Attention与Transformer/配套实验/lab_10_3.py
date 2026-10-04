"""Runnable examples for lesson 10.3."""

import statistics
import time
import torch
from transformer_core import TinyTransformer

torch.manual_seed(35)
torch.set_num_threads(1)
V, D, F, L, M, h = 9, 32, 64, 2, 128, 4
model = TinyTransformer(vocab=V, width=D, heads=h, ff_width=F,
                        depth=L, max_length=M).eval()
formula = 2*V*D + M*D + L*(4*D*D + 2*D*F + 9*D + F) + 2*D
actual = sum(p.numel() for p in model.parameters())
assert formula == actual
print("exact trainable parameters:", actual)
print("FP32 parameter bytes:", sum(p.numel()*p.element_size() for p in model.parameters()))
print("rough FP32 Adam persistent bytes:", 16 * actual)

resource_rows = []
with torch.inference_mode():
    for T in [8, 16, 32, 64]:
        B = 4
        ids = torch.randint(1, V, (B, T))
        for _ in range(4):
            model(ids)
        elapsed = []
        for _ in range(7):
            start = time.perf_counter()
            for _ in range(10):
                model(ids)
            elapsed.append((time.perf_counter() - start) / 10)
        seconds = statistics.median(elapsed)
        dense_map_bytes = B * h * T * T * 4
        kv_bytes = 2 * L * B * T * h * (D // h) * 4
        forward_macs = L * (4*B*T*D*D + 2*B*T*F*D + 2*B*T*T*D) + B*T*D*V
        resource_rows.append((T, seconds*1000, B*T/seconds,
                              dense_map_bytes, kv_bytes, forward_macs))
print("T / ms / input tokens per second / map bytes per layer / KV bytes / forward MACs")
for row in resource_rows:
    print(row)

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
lengths = [r[0] for r in resource_rows]
axes[0].plot(lengths, [r[3]/1024 for r in resource_rows], "o-",
             label="One dense map per layer")
axes[0].plot(lengths, [r[4]/1024 for r in resource_rows], "o-",
             label="All-layer KV cache")
axes[0].set(xlabel="Sequence length", ylabel="Logical KiB (different workloads)",
            title="Tensor estimates, not measured peak memory")
axes[0].legend(fontsize=8)
axes[1].plot(lengths, [r[1] for r in resource_rows], "o-")
axes[1].set(xlabel="Sequence length", ylabel="Median full forward latency (ms)",
            title="Local CPU timing, batch size 4")
fig.tight_layout()
fig.savefig(output_dir / "10_3_resources.png", dpi=150)
plt.close(fig)
