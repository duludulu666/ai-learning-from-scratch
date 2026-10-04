import torch
from torch import nn
import torch.nn.functional as F
from multimodal_core import setup, bars

def multipos_loss(scores, positive):
    if scores.ndim != 2 or scores.numel() == 0 or positive.shape != scores.shape:
        raise ValueError("Need nonempty scores and a same-shape positive mask")
    if positive.dtype != torch.bool or not torch.isfinite(scores).all():
        raise ValueError("Need finite scores and a boolean positive mask")
    if not positive.any(1).all():
        raise ValueError("Every anchor needs at least one positive")
    return (scores.logsumexp(1) - scores.masked_fill(~positive, -torch.inf).logsumexp(1)).mean()

def main():
    setup(152)
    s = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
    y = torch.arange(2)
    loss = (F.cross_entropy(s, y) + F.cross_entropy(s.T, y)) / 2
    loss.backward()
    expected = ((s.detach().softmax(1) - torch.eye(2)) +
                (s.detach().softmax(0) - torch.eye(2))) / 4
    torch.testing.assert_close(s.grad, expected)

    # General non-symmetric scores and complete raw-feature gradient.
    gen = torch.Generator().manual_seed(1520)
    raw_v = torch.randn(3,4,generator=gen,dtype=torch.float64,requires_grad=True)
    raw_u = torch.randn(3,4,generator=gen,dtype=torch.float64,requires_grad=True)
    v, u = F.normalize(raw_v,dim=-1), F.normalize(raw_u,dim=-1)
    scores = v@u.T/.3
    objective = (F.cross_entropy(scores,torch.arange(3))+
                 F.cross_entropy(scores.T,torch.arange(3)))/2
    gv,gu = torch.autograd.grad(objective,(raw_v,raw_u))
    ds = (scores.softmax(1)+scores.softmax(0)-2*torch.eye(3,dtype=torch.float64))/6
    dv,du = ds@u/.3, ds.T@v/.3
    manual_v = (dv-v*(dv*v).sum(-1,keepdim=True))/raw_v.norm(dim=-1,keepdim=True)
    manual_u = (du-u*(du*u).sum(-1,keepdim=True))/raw_u.norm(dim=-1,keepdim=True)
    torch.testing.assert_close(gv,manual_v)
    torch.testing.assert_close(gu,manual_u)
    sample = torch.tensor([[2.,1.,0.]],dtype=torch.float64,requires_grad=True)
    mask = torch.tensor([[True,True,False]])
    multi = multipos_loss(sample,mask)
    expected = sample.softmax(1)-sample.masked_fill(~mask,-torch.inf).softmax(1)
    torch.testing.assert_close(torch.autograd.grad(multi,sample)[0],expected)
    for invalid_scores, invalid_mask in [
        (torch.zeros(1,3),torch.zeros(1,3,dtype=torch.bool)),
        (torch.zeros(1,3),torch.ones(1,3)),
        (torch.full((1,3),float("nan")),torch.ones(1,3,dtype=torch.bool))
    ]:
        try:
            multipos_loss(invalid_scores,invalid_mask)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid contrastive inputs accepted")
    x, labels = bars(64, 31)

    tx, ty = bars(64, 32)
    vision = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 8))
    text = nn.Embedding(2, 8)
    opt = torch.optim.Adam(list(vision.parameters()) + list(text.parameters()), lr=0.01)
    positive = labels[:, None] == labels[None, :]
    for _ in range(100):
        v = F.normalize(vision(x), dim=-1)
        t = F.normalize(text(labels), dim=-1)
        scores = v @ t.T / 0.2
        loss = (multipos_loss(scores, positive) + multipos_loss(scores.T, positive.T)) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        prototypes = F.normalize(text(torch.arange(2)), dim=-1)
        logits = F.normalize(vision(tx), dim=-1) @ prototypes.T / 0.2
        acc = (logits.argmax(1) == ty).float().mean().item()
    identical = torch.zeros(4, 4)
    naive = F.cross_entropy(identical, torch.arange(4))
    grouped = multipos_loss(identical, torch.ones(4, 4, dtype=torch.bool))
    assert abs(naive.item() - torch.log(torch.tensor(4.)).item()) < 1e-6
    assert grouped.item() == 0
    print(f"contrastive gradient passed; held-out image class accuracy={acc:.4f}")
    print(f"all-same-class fixture: diagonal loss={naive:.4f}, grouped loss={grouped:.4f}")
    print("Two learned text IDs, not a natural-language text encoder or unseen-class zero-shot.")

if __name__ == "__main__":
    main()
