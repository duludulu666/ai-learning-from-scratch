"""Runnable examples for lesson 09.4."""

from transformer_core import TinyTransformer

import copy
import torch
import torch.nn.functional as F

torch.manual_seed(30)
torch.set_num_threads(1)
generator = torch.Generator().manual_seed(31)
alphabet = torch.arange(1, 9)
all_prompts = torch.cartesian_prod(alphabet, alphabet, alphabet, alphabet)
order = torch.randperm(len(all_prompts), generator=generator)
train_prompt = all_prompts[order[:512]]
val_prompt = all_prompts[order[512:640]]
test_prompt = all_prompts[order[640:768]]
split_sets = [set(map(tuple, part.tolist()))
              for part in (train_prompt, val_prompt, test_prompt)]
assert all(split_sets[i].isdisjoint(split_sets[j])
           for i in range(3) for j in range(i + 1, 3))
batch_rng_state = generator.get_state()

def supervised_batch(prompts):
    full = torch.cat([prompts, prompts], dim=1)
    return full[:, :-1], full[:, 1:]

model = TinyTransformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)

@torch.no_grad()
def evaluate_copy(prompts):
    model.eval()
    inputs, targets = supervised_batch(prompts)
    logits = model(inputs)[:, 3:]
    labels = targets[:, 3:]
    loss = F.cross_entropy(logits.reshape(-1, 9), labels.reshape(-1)).item()
    accuracy = logits.argmax(-1).eq(labels).float().mean().item()
    return loss, accuracy

history = []
initial_val = evaluate_copy(val_prompt)[0]
best_loss, best_state = float("inf"), None
for step in range(600):
    model.train()
    indices = torch.randint(len(train_prompt), (64,), generator=generator)
    inputs, targets = supervised_batch(train_prompt[indices])
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)[:, 3:]
    loss = F.cross_entropy(logits.reshape(-1, 9), targets[:, 3:].reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm)
    optimizer.step()
    if (step + 1) % 25 == 0:
        train_loss, _ = evaluate_copy(train_prompt)
        val_loss, val_accuracy = evaluate_copy(val_prompt)
        history.append((step + 1, train_loss, val_loss, val_accuracy))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    # Perturb future inputs; earlier logits must not change.
    inputs, _ = supervised_batch(test_prompt[:4])
    changed = inputs.clone()
    changed[:, 4:] = (changed[:, 4:] % 8) + 1
    torch.testing.assert_close(model(inputs)[:, :4], model(changed)[:, :4],
                               atol=1e-6, rtol=1e-5)
    generated = model.generate(test_prompt, new_tokens=4)[:, 4:]
    exact_match = generated.eq(test_prompt).all(dim=1).float().mean().item()

print("parameter count:", sum(p.numel() for p in model.parameters()))
print("initial / best validation loss:", initial_val, best_loss)
test_metrics = evaluate_copy(test_prompt)
print("teacher-forced test loss / accuracy:", test_metrics)
print("free-running exact match:", exact_match)
print("prompt / generated:", test_prompt[:5], generated[:5])
# A regression guard for this fixed synthetic setup, not a general quality bar.
assert best_loss < initial_val * 0.1
assert test_metrics[1] > 0.9 and exact_match > 0.8

# Separately trained no-context baseline: token/position features and FFNs only.
class PositionwiseBaseline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token = torch.nn.Embedding(9, 32)
        self.position = torch.nn.Embedding(16, 32)
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.LayerNorm(32), torch.nn.Linear(32, 64),
                                torch.nn.GELU(), torch.nn.Linear(64, 32))
            for _ in range(2)
        ])
        self.final_norm = torch.nn.LayerNorm(32)
        self.head = torch.nn.Linear(32, 9, bias=False)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.position(positions)[None]
        for block in self.blocks:
            x = x + block(x)
        return self.head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, prompt, new_tokens):
        self.eval()
        result = prompt.clone()
        for _ in range(new_tokens):
            next_id = self(result)[:, -1].argmax(-1, keepdim=True)
            result = torch.cat([result, next_id], dim=1)
        return result

torch.manual_seed(130)
baseline = PositionwiseBaseline()
baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.003, weight_decay=0.01)
baseline_generator = torch.Generator()
baseline_generator.set_state(batch_rng_state)

@torch.no_grad()
def baseline_evaluate(prompts):
    baseline.eval()
    inputs, targets = supervised_batch(prompts)
    logits, labels = baseline(inputs)[:, 3:], targets[:, 3:]
    return (F.cross_entropy(logits.reshape(-1, 9), labels.reshape(-1)).item(),
            logits.argmax(-1).eq(labels).float().mean().item())

baseline_history = []
baseline_best_loss, baseline_best_state = float("inf"), None
for step in range(600):
    baseline.train()
    indices = torch.randint(len(train_prompt), (64,), generator=baseline_generator)
    inputs, targets = supervised_batch(train_prompt[indices])
    baseline_optimizer.zero_grad(set_to_none=True)
    logits = baseline(inputs)[:, 3:]
    loss = F.cross_entropy(logits.reshape(-1, 9), targets[:, 3:].reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
    assert torch.isfinite(grad_norm)
    baseline_optimizer.step()
    if (step + 1) % 25 == 0:
        train_loss, _ = baseline_evaluate(train_prompt)
        val_loss, val_accuracy = baseline_evaluate(val_prompt)
        baseline_history.append((step + 1, train_loss, val_loss, val_accuracy))
        if val_loss < baseline_best_loss:
            baseline_best_loss = val_loss
            baseline_best_state = copy.deepcopy(baseline.state_dict())

baseline.load_state_dict(baseline_best_state)
baseline.eval()
baseline_test = baseline_evaluate(test_prompt)
baseline_generated = baseline.generate(test_prompt, 4)[:, 4:]
baseline_exact = baseline_generated.eq(test_prompt).all(1).float().mean().item()
# Verify the baseline actually cannot read previous positions.
with torch.no_grad():
    unchanged_last = test_prompt[:4].clone()
    changed_past = unchanged_last.clone()
    changed_past[:, :-1] = (changed_past[:, :-1] % 8) + 1
    torch.testing.assert_close(baseline(unchanged_last)[:, -1],
                               baseline(changed_past)[:, -1])
print("baseline parameter count:", sum(p.numel() for p in baseline.parameters()))
print("baseline best validation / test loss+accuracy / exact:",
      baseline_best_loss, baseline_test, baseline_exact)

# Exhaustive population check: each (input position, current symbol) has
# a uniform target distribution. This verifies the 1/8 information bound.
population_input, population_target = supervised_batch(all_prompts)
for t in range(3, 7):
    for symbol in range(1, 9):
        labels = population_target[population_input[:, t].eq(symbol), t]
        counts = torch.bincount(labels, minlength=9)[1:]
        assert torch.equal(counts, torch.full_like(counts, 64))
print("full-population no-context teacher-forced accuracy bound = 1/8")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
steps = [r[0] for r in history]
axes[0].semilogy(steps, [r[1] for r in history], label="Train copy loss")
axes[0].semilogy(steps, [r[2] for r in history], label="Validation copy loss")
axes[0].semilogy(steps, [r[2] for r in baseline_history], "--", label="No-context validation loss")
axes[0].set(xlabel="Optimizer update", ylabel="Cross-entropy", title="Delayed-copy task")
axes[0].legend()
axes[1].plot(steps, [r[3] for r in history], label="Transformer")
axes[1].plot(steps, [r[3] for r in baseline_history], "--", label="No-context baseline")
axes[1].axhline(1 / 8, color="gray", linewidth=1, linestyle=":", label="Population chance = 1/8")
axes[1].legend(fontsize=8)
axes[1].set(xlabel="Optimizer update", ylabel="Validation token accuracy", ylim=(0, 1.05),
            title="Teacher-forced evaluation")
fig.tight_layout()
fig.savefig(output_dir / "09_4_learning.png", dpi=150)
plt.close(fig)
