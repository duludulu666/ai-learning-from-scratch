import torch
from multimodal_core import setup, bars, TinyVLM, answer_loss

def main():
    setup(154)
    x, y = bars(96, 41)
    tx, ty = bars(64, 42)
    model = TinyVLM()
    ids = torch.stack([torch.ones_like(y), torch.full_like(y, 2), y + 3], dim=1)
    targets = torch.stack([torch.full_like(y, -100), y + 3, torch.full_like(y, 5)], dim=1)
    # BOS predicts the prompt (ignored); QUESTION predicts answer; answer predicts EOS.
    opt = torch.optim.Adam(model.parameters(), lr=0.004)
    losses = []
    for _ in range(220):
        logits = model(x, ids)
        logits.retain_grad()
        loss = answer_loss(logits, targets)
        opt.zero_grad()
        loss.backward()
        assert logits.grad[:, 0].abs().max() == 0
        assert model.vision.patch.weight.grad.abs().sum() > 0
        opt.step()
        losses.append(loss.item())
    model.eval()
    prompt = torch.tensor([[1, 2]]).expand(len(tx), -1)
    with torch.no_grad():
        ans = model(tx, prompt)[:, -1].argmax(-1)
        next_ids = torch.cat([prompt, ans[:, None]], dim=1)
        eos = model(tx, next_ids)[:, -1].argmax(-1)
        acc = ((ans == ty + 3) & (eos == 5)).float().mean().item()
        blind = model(torch.zeros_like(tx), prompt)[:, -1].argmax(-1)
        blind_acc = (blind == ty + 3).float().mean().item()
        swapped = model(tx.flip(0), prompt)[:, -1].argmax(-1)
        swapped_old = (swapped == ty + 3).float().mean().item()
        swapped_new = (swapped == ty.flip(0) + 3).float().mean().item()
        perm = torch.randperm(len(tx),generator=torch.Generator().manual_seed(1541))
        # Batch permutation must not provide the alternating-label answer.
        permuted = model(tx[perm],prompt)[:,-1].argmax(-1)
        torch.testing.assert_close(permuted,ans[perm])
        # Future teacher-forcing answer must not change earlier text logits.
        a = model(tx[:2], torch.tensor([[1, 2, 3], [1, 2, 3]]))
        b = model(tx[:2], torch.tensor([[1, 2, 4], [1, 2, 4]]))
        torch.testing.assert_close(a[:, :2], b[:, :2])

    assert sum(p.numel() for p in model.parameters()) == 6214
    # Ignored nonfinite output leaves must not contaminate active CE.
    leaves = torch.randn(2,3,6)
    leaves[:,0] = float("nan")
    leaves.requires_grad_()
    target = torch.tensor([[-100,3,5],[-100,4,5]])
    answer_loss(leaves,target).backward()
    assert torch.isfinite(leaves.grad).all() and leaves.grad[:,0].abs().sum()==0
    for bad_ids in [torch.zeros(2,2,dtype=torch.long),torch.ones(2,6,dtype=torch.long)]:
        try:
            model(tx[:2],bad_ids)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported PAD/context accepted")

    for bad_logits,bad_targets in [
        (torch.randn(2,3,6),torch.ones(2,2,dtype=torch.long)),
        (torch.randn(2,3,6),torch.ones(2,3)),
        (torch.randn(2,3,6),torch.full((2,3),6,dtype=torch.long)),
        (torch.full((2,3,6),float("nan")),torch.ones(2,3,dtype=torch.long))
    ]:
        try:
            answer_loss(bad_logits,bad_targets)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid active response loss accepted")
    assert losses[-1] < losses[0]


    try:
        answer_loss(torch.randn(2, 3, 6), torch.full((2, 3), -100))
    except ValueError:
        pass
    else:
        raise AssertionError("Empty response should be rejected")
    print(f"held-out free generation exact match={acc:.4f}; blank-image={blind_acc:.4f}")
    print(f"swapped image scored against ORIGINAL label={swapped_old:.4f}; final loss={losses[-1]:.4f}")
    print(f"swapped image scored against NEW label={swapped_new:.4f}; batch-permutation equivariance passed")
    print("Jointly trained miniature conditional decoder, not pretrained LLM instruction tuning.")

if __name__ == "__main__":
    main()
