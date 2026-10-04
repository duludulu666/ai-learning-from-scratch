"""Causal loss, token-weighted perplexity, and boundary conditions."""
import math
import torch
from llm_core import causal_batch, token_nll, BOS, EOS, PAD, seed_all

seed_all()
sequences = [[BOS,5,8,EOS], [BOS,6,EOS]]
x, y, valid = causal_batch(sequences)
assert x.tolist() == [[BOS,5,8], [BOS,6,EOS]]
assert y.tolist() == [[5,8,EOS], [6,EOS,PAD]]
assert valid.sum().item() == 5
logits = torch.randn(2,3,10,dtype=torch.float64,requires_grad=True)
total, count = token_nll(logits,y,valid)
manual = -logits.log_softmax(-1).gather(-1,y[...,None]).squeeze(-1)[valid].sum()
torch.testing.assert_close(total,manual)
loss = total/count
loss.backward()
expected = logits.detach().softmax(-1)
expected.scatter_add_(-1,y[...,None],-torch.ones_like(y[...,None],dtype=torch.float64))
expected *= valid[...,None]/count
torch.testing.assert_close(logits.grad,expected)
changed_logits = logits.detach().clone()
changed_targets = y.clone()
changed_logits[~valid] = float("nan")
changed_targets[~valid] = -999
torch.testing.assert_close(token_nll(changed_logits,changed_targets,valid)[0],total.detach())
try:
    token_nll(logits,y,torch.zeros_like(valid))
    raise AssertionError("empty supervision accepted")
except ValueError:
    pass
probabilities = torch.tensor([.5,.25,.5],dtype=torch.float64)
mean_nll = float(-probabilities.log().mean())
assert math.isclose(math.exp(mean_nll),16**(1/3))
group_sums = torch.tensor([2.,18.])
counts = torch.tensor([2.,6.])
assert float(group_sums.sum()/counts.sum()) == 2.5
assert float((group_sums/counts).mean()) == 2.
print("NLL / perplexity:", mean_nll, math.exp(mean_nll))
print("unequal groups: correct mean 2.5; mean-of-means 2.0")
print("shift, selected CE, full logit gradient, ignored NaN, and empty-mask tests passed")


# Log-sum-exp stability and output-head gradients on a masked batch.
extreme=torch.tensor([[10000.,9999.,-10000.]],dtype=torch.float64)
label=torch.tensor([1]);mask_one=torch.ones(1,dtype=torch.bool)
stable,_=token_nll(extreme,label,mask_one)
shifted,_=token_nll(extreme-10000,label,mask_one)
torch.testing.assert_close(stable,shifted)
H=torch.arange(12,dtype=torch.float64).reshape(2,3,2).requires_grad_()
W=torch.linspace(-.4,.5,8,dtype=torch.float64).reshape(4,2).requires_grad_()
bias=torch.zeros(4,dtype=torch.float64,requires_grad=True)
head_targets=torch.tensor([[0,1,2],[3,0,-100]])
head_mask=head_targets.ne(-100)
Z=H@W.T+bias
total,count=token_nll(Z,head_targets,head_mask)
(total/count).backward()
G=torch.zeros_like(Z)
G[head_mask]=Z.detach()[head_mask].softmax(-1)
indices=head_mask.nonzero()
G[indices[:,0],indices[:,1],head_targets[head_mask]]-=1
G/=count
torch.testing.assert_close(W.grad,G.reshape(-1,4).T@H.detach().reshape(-1,2))
torch.testing.assert_close(bias.grad,G.sum((0,1)))
torch.testing.assert_close(H.grad,G@W.detach())
for bad in [torch.tensor([4]),torch.tensor([-1]),torch.tensor([1.])]:
    try:
        token_nll(torch.zeros(1,4),bad,mask_one)
        raise AssertionError("invalid supervised target accepted")
    except ValueError:
        pass
print("logit-shift stability, vocabulary-head weight/bias/input gradients and invalid targets passed")
