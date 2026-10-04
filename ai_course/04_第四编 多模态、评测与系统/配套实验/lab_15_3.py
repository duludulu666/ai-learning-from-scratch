import torch
from torch import nn
from multimodal_core import setup


def safe_cross_attention(layer, text, visual, blocked):
    if blocked.dtype != torch.bool or blocked.shape != visual.shape[:2]:
        raise ValueError("Expected a boolean visual padding mask (B,N)")
    if not (~blocked).any(1).all():
        raise ValueError("An example has no available visual token")
    # Masks are not sanitizers: QK is computed before the score mask.
    if not torch.isfinite(text).all() or not torch.isfinite(visual).all():
        raise ValueError("Nonfinite text/visual features are not supported")
    return layer(text,visual,visual,key_padding_mask=blocked)

def main():

    setup(153)
    attn = nn.MultiheadAttention(8, 2, dropout=0, batch_first=True).eval()
    text = torch.randn(2, 3, 8, requires_grad=True)
    image = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[False, False, False, True, True]]).expand(2, -1)
    out, weights = safe_cross_attention(attn,text,image,mask)
    changed = image.detach().clone()
    changed[:, 3:] += 1000
    out2, _ = attn(text, changed, changed, key_padding_mask=mask)
    torch.testing.assert_close(out, out2)
    out.square().mean().backward()
    assert image.grad[:, :3].abs().sum() > 0
    assert image.grad[:, 3:].abs().max() == 0
    assert out.shape == (2, 3, 8) and weights.shape == (2, 3, 5)
    # Jointly permuting keys, values, and padding mask preserves the readout.
    perm = torch.tensor([4, 2, 0, 3, 1])
    out3, _ = attn(text.detach(), image.detach()[:, perm], image.detach()[:, perm],
                   key_padding_mask=mask[:, perm])
    torch.testing.assert_close(out.detach(), out3)

    for visual,blocked in [
        (image.detach(),torch.ones_like(mask)),
        (image.detach().masked_fill(mask[:,:,None],float("nan")),mask)
    ]:
        try:
            safe_cross_attention(attn,text.detach(),visual,blocked)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid visual evidence accepted")
    # Manual unprojected single-head backward on a non-square T x N score matrix.
    q = torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    k = torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    v = torch.randn(4,2,dtype=torch.float64,requires_grad=True)
    blocked = torch.tensor([[False,False,False,True]]).expand(2,-1)
    a = (q@k.T/(3**.5)).masked_fill(blocked,-torch.inf).softmax(1)
    o = a@v
    go = torch.randn_like(o)
    actual = torch.autograd.grad((o*go).sum(),(q,k,v))
    ga = go@v.T
    gs = a*(ga-(a*ga).sum(1,keepdim=True))
    manual = (gs@k/(3**.5), gs.T@q/(3**.5), a.T@go)
    for expected,observed in zip(manual,actual):
        torch.testing.assert_close(expected,observed)


    # Frozen downstream weights can still propagate input gradients to a connector.
    frozen = nn.Linear(3,2)
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    connector_output = torch.randn(2,3,requires_grad=True)
    frozen(connector_output).square().mean().backward()
    assert connector_output.grad.abs().sum()>0
    assert all(p.grad is None for p in frozen.parameters())
    with torch.no_grad():
        detached = frozen(connector_output)
    assert not detached.requires_grad
    print("cross-attention shapes, padding isolation, manual gradients, frozen-input path and permutation passed")


if __name__ == "__main__":
    main()
