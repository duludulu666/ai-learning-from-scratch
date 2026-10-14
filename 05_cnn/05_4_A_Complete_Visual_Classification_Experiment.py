"""05.4 A Complete Visual Classification Experiment.

End-to-end pipeline on generated 16x16 grayscale images:
label 0 = horizontal bright stripe, label 1 = vertical bright stripe.
Compares MLP vs CNN vs CNN+BN, selects the best on validation, then
evaluates the frozen choice on test (and a separate noisier shift set).
No external dataset or GPU needed.
"""

import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(1)


# ---------- 1. Generate synthetic images: horizontal vs vertical stripe ----------
def make_images(n, seed, noise=0.15):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 2
    images = noise * torch.randn(n, 1, 16, 16, generator=generator)
    for i in range(n):
        position = int(torch.randint(2, 12, (), generator=generator))
        width = int(torch.randint(1, 4, (), generator=generator))
        if labels[i] == 0:
            images[i, 0, position:position+width, :] += 1.0   # horizontal stripe
        else:
            images[i, 0, :, position:position+width] += 1.0    # vertical stripe
    return images, labels.long()


train_x, train_y = make_images(256, 1)
val_x, val_y = make_images(96, 2)
test_x, test_y = make_images(96, 3)
shift_x, shift_y = make_images(96, 4, noise=0.45)   # separate robustness test

# Normalize using TRAIN statistics only (no leakage from val/test)
mean = train_x.mean()
std = train_x.std(unbiased=False).clamp_min(1e-8)
train_x, val_x = (train_x - mean) / std, (val_x - mean) / std
test_x, shift_x = (test_x - mean) / std, (shift_x - mean) / std


# ---------- 2. Three models: MLP, CNN, CNN+BN ----------
def make_model(name):
    if name == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(256, 32),
                             nn.ReLU(), nn.Linear(32, 2))
    use_bn = name == "cnn_bn"
    def norm(channels):
        return nn.BatchNorm2d(channels) if use_bn else nn.Identity()
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), norm(8), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), norm(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 2),
    )


# ---------- 3. Evaluation: loss, accuracy, macro F1, confusion matrix ----------
@torch.no_grad()
def evaluate(model, images, labels):
    model.eval()
    logits = model(images)
    loss = F.cross_entropy(logits, labels).item()
    predicted = logits.argmax(1)
    confusion = torch.bincount(2 * labels + predicted, minlength=4).reshape(2, 2)
    accuracy = predicted.eq(labels).float().mean().item()
    f1_scores = []
    for cls in range(2):
        tp = confusion[cls, cls].item()
        fp = confusion[:, cls].sum().item() - tp
        fn = confusion[cls, :].sum().item() - tp
        f1_scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return loss, accuracy, sum(f1_scores) / 2, confusion


# Verify shared init: cnn and cnn_bn must start from identical conv/head weights
torch.manual_seed(10)
plain_initial = make_model("cnn").state_dict()
torch.manual_seed(10)
bn_initial = make_model("cnn_bn").state_dict()
for key in plain_initial.keys() & bn_initial.keys():
    torch.testing.assert_close(plain_initial[key], bn_initial[key], atol=0, rtol=0)
batch_orders = {}


# ---------- 4. Train all three models with identical data order ----------
models, histories, validation_losses = {}, {}, {}
for name in ["mlp", "cnn", "cnn_bn"]:
    torch.manual_seed(10)
    model = make_model(name)
    loader = DataLoader(TensorDataset(train_x, train_y, torch.arange(len(train_y))),
                        batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(20))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    best_loss, best_state = float("inf"), None
    history, seen_indices = [], []

    for epoch in range(20):
        model.train()
        for images, labels, example_indices in loader:
            seen_indices.append(example_indices.clone())
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        train_result = evaluate(model, train_x, train_y)
        val_result = evaluate(model, val_x, val_y)
        history.append((train_result[0], val_result[0]))
        if val_result[0] < best_loss:           # checkpoint selection by val loss
            best_loss = val_result[0]
            best_state = copy.deepcopy(model.state_dict())

    batch_orders[name] = torch.cat(seen_indices)
    model.load_state_dict(best_state)
    models[name], histories[name], validation_losses[name] = model, history, best_loss
    print(name, "parameters:", sum(p.numel() for p in model.parameters()),
          "best validation:", evaluate(model, val_x, val_y)[:3])

# Sanity: all three models saw identical batch orders
torch.testing.assert_close(batch_orders['cnn'], batch_orders['cnn_bn'], atol=0, rtol=0)
torch.testing.assert_close(batch_orders['mlp'], batch_orders['cnn'], atol=0, rtol=0)


# ---------- 5. Select on validation, freeze, then evaluate on test ----------
selected = min(validation_losses, key=validation_losses.get)
final_model = models[selected]
print("selected using validation:", selected)
frozen_state = {k: v.clone() for k, v in final_model.state_dict().items()}
test_result = evaluate(final_model, test_x, test_y)
shift_result = evaluate(final_model, shift_x, shift_y)

# Confirm evaluation didn't accidentally mutate the model (train-mode bug check)
for key, value in final_model.state_dict().items():
    torch.testing.assert_close(value, frozen_state[key], atol=0, rtol=0)

assert test_result[3].sum().item() == len(test_y)
assert test_result[1] > .9   # regression guard for this fixed teaching task
print("test:", test_result)
print("noise-shift test:", shift_result)
print("shared CNN initialization and frozen evaluation verified")
