import torch
from torch import nn
import torch.nn.functional as F
from multimodal_core import setup


def fuse_probabilities(prob, available, prior_weights):
    if prob.ndim != 2 or available.shape != prob.shape or available.dtype != torch.bool:
        raise ValueError("Need (B,M) probabilities and boolean availability")
    if prior_weights.shape != (prob.shape[1],) or not torch.isfinite(prior_weights).all() or not (prior_weights>0).all():
        raise ValueError("Need finite positive branch weights")
    # Unavailable values may be NaN; select a safe value BEFORE multiplication.
    clean = torch.where(available,prob,torch.zeros_like(prob))
    if not torch.isfinite(clean).all() or not ((clean>=0)&(clean<=1)).all():
        raise ValueError("Available branch probabilities must be finite and in [0,1]")
    weights = available*prior_weights
    if not (weights.sum(1)>0).all():
        raise ValueError("No available modality")
    return (weights*clean).sum(1)/weights.sum(1)

def main():

    setup(162)
    x = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y = torch.tensor([0.,1.,1.,0.])
    early = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    # Late additive logits with independent binary branch lookups.
    branch_a, branch_b = nn.Embedding(2, 1), nn.Embedding(2, 1)
    opt_e = torch.optim.Adam(early.parameters(), lr=.03)
    opt_l = torch.optim.Adam(list(branch_a.parameters()) + list(branch_b.parameters()), lr=.03)
    for _ in range(700):
        e = early(x).squeeze(-1)
        l = (branch_a(x[:, 0].long()) + branch_b(x[:, 1].long())).squeeze(-1)
        for opt, logits in [(opt_e, e), (opt_l, l)]:
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    ep = early(x).squeeze().sigmoid().detach()
    lp = (branch_a(x[:, 0].long())+branch_b(x[:, 1].long())).squeeze().sigmoid().detach()
    assert (ep.round() == y).all()
    assert F.binary_cross_entropy(lp, y) > .68
    # Availability-renormalized probability fusion, not a learned robustness guarantee.
    prob = torch.tensor([[.9,.2],[.9,.2],[.9,.2]])
    avail = torch.tensor([[1.,1.],[1.,0.],[0.,1.]])
    prior_weights = torch.tensor([.6,.4])
    fused = fuse_probabilities(prob,avail.bool(),prior_weights)
    torch.testing.assert_close(fused, torch.tensor([.62,.9,.2]))

    missing = prob.clone()
    missing[1,1] = float("nan")
    torch.testing.assert_close(fuse_probabilities(missing,avail.bool(),prior_weights),fused)
    try:
        fuse_probabilities(prob[:1],torch.zeros(1,2,dtype=torch.bool),prior_weights)
    except ValueError:
        pass
    else:
        raise AssertionError("All-missing inputs were accepted")
    p1,p2 = torch.tensor([.9,.1]),torch.tensor([.6,.4])
    mixture = (p1+p2)/2
    averaged_logits = ((p1.log()+p2.log())/2).softmax(0)
    summed_logits = (p1.log()+p2.log()).softmax(0)
    torch.testing.assert_close(averaged_logits,(p1*p2).sqrt()/(p1*p2).sqrt().sum())
    torch.testing.assert_close(summed_logits,p1*p2/(p1*p2).sum())
    assert not torch.allclose(mixture,averaged_logits)
    print("probability mixture / logit average / logit sum:",
          mixture[0].item(),averaged_logits[0].item(),summed_logits[0].item())

    print("XOR early probabilities:", ep.tolist())
    print("XOR additive late probabilities:", lp.tolist())
    print("availability-renormalized fusion:", fused.tolist())
    print("Truth-table capacity test, not a held-out generalization result.")

if __name__ == "__main__":
    main()
