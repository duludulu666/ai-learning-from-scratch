import torch
from torch import nn
import torch.nn.functional as F
from multimodal_core import setup, bars, TinyViT, save_figure

def main():
    setup(151)
    x = torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    conv = nn.Conv2d(1, 6, 2, stride=2)
    patches = F.unfold(x, kernel_size=2, stride=2).transpose(1, 2)
    direct = F.linear(patches, conv.weight.flatten(1), conv.bias)
    torch.testing.assert_close(direct, conv(x).flatten(2).transpose(1, 2))
    assert patches.shape == (2, 16, 4)

    # Compare backward paths too, using identical incoming output sensitivities.
    check_x = torch.randn(2, 3, 4, 4, dtype=torch.float64, requires_grad=True)
    check_w = torch.randn(5, 3, 2, 2, dtype=torch.float64, requires_grad=True)
    check_b = torch.randn(5, dtype=torch.float64, requires_grad=True)
    via_conv = F.conv2d(check_x, check_w, check_b, stride=2).flatten(2).transpose(1,2)
    via_linear = F.linear(F.unfold(check_x, 2, stride=2).transpose(1,2),
                          check_w.flatten(1), check_b)
    incoming = torch.randn_like(via_conv)
    grad_conv = torch.autograd.grad((via_conv*incoming).sum(),(check_x,check_w,check_b))
    grad_linear = torch.autograd.grad((via_linear*incoming).sum(),(check_x,check_w,check_b))
    for a,b in zip(grad_conv,grad_linear):
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    model = TinyViT()
    assert sum(p.numel() for p in model.parameters()) == 2626
    # No positions + mean pooling: encoded-token permutations do not change readout.
    with torch.no_grad():
        token = torch.randn(2,16,16)
        permutation = torch.arange(15,-1,-1)
        one = model.norm(model.encoder(token)).mean(1)
        two = model.norm(model.encoder(token[:,permutation])).mean(1)
        torch.testing.assert_close(one,two,atol=1e-6,rtol=1e-5)
    try:
        model(torch.randn(2,1,8,9))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid image width was silently truncated")
    setup(151)  # Preserve the original training initialization after auxiliary tests.
    _ = nn.Conv2d(1,6,2,stride=2)
    model = TinyViT()

    train_x, train_y = bars(96, 1)
    test_x, test_y = bars(64, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.006)
    losses = []
    for _ in range(160):
        opt.zero_grad()
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        acc = (model(test_x).argmax(-1) == test_y).float().mean().item()
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
    print(f"patch equivalence passed; test accuracy={acc:.4f}; loss={losses[-1]:.4f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, i in zip(axes[:2], [0, 1]):
        ax.imshow(train_x[i, 0], cmap="gray")
        ax.set(title=["Horizontal", "Vertical"][i], xticks=[], yticks=[])
        for k in [1.5, 3.5, 5.5]:
            ax.axvline(k, color="tab:orange", lw=1)
            ax.axhline(k, color="tab:orange", lw=1)
    axes[2].plot(losses)
    axes[2].set(xlabel="Update", ylabel="Training CE", title="Tiny ViT, synthetic bars")
    fig.tight_layout()
    save_figure(fig, "15_patch_and_training.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
