"""DPO on categorical candidate policies; token-sum fixture and manual derivatives."""
import torch
import torch.nn.functional as F
from llm_core import seed_all

seed_all()
beta=.3
margin=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
loss=F.softplus(-beta*margin)
loss.backward()
expected=-beta*torch.sigmoid(-beta*margin.detach())
torch.testing.assert_close(margin.grad,expected)
eps=1e-6
numerical=(F.softplus(-beta*(margin.detach()+eps))-F.softplus(-beta*(margin.detach()-eps)))/(2*eps)
torch.testing.assert_close(margin.grad,numerical)
policy=torch.nn.Parameter(torch.zeros(2,3,dtype=torch.float64))
reference=torch.log_softmax(torch.tensor([[.2,0.,-.2],[-.1,.2,0.]],dtype=torch.float64),-1)
initial=policy.detach().softmax(-1).clone()
optimizer=torch.optim.SGD([policy],lr=.4)
# Prompt 0 prefers response 0 over 1; prompt 1 prefers response 2 over 0.
winner=torch.tensor([0,2]);loser=torch.tensor([1,0]);rows=torch.arange(2)
for _ in range(150):
    logp=policy.log_softmax(-1)
    delta=(logp[rows,winner]-reference[rows,winner])-(logp[rows,loser]-reference[rows,loser])
    objective=F.softplus(-beta*delta).mean()
    optimizer.zero_grad(set_to_none=True);objective.backward();optimizer.step()
final=policy.detach().softmax(-1)
assert (final[rows,winner]>initial[rows,winner]).all()
print("initial/final probabilities:",initial.tolist(),final.tolist())
with torch.no_grad():
    logp=policy.log_softmax(-1)
    final_delta=(logp[rows,winner]-reference[rows,winner])-(logp[rows,loser]-reference[rows,loser])
    final_objective=F.softplus(-beta*final_delta).mean()
print("final preference loss:",float(final_objective))
# Response log probabilities are sums, not token means.
logits=torch.randn(2,4,5,dtype=torch.float64,requires_grad=True)
targets=torch.tensor([[1,2,3,0],[3,1,0,0]])
mask=torch.tensor([[1,1,1,0],[1,1,0,0]],dtype=torch.bool)
logprob=logits.log_softmax(-1).gather(-1,targets[...,None]).squeeze(-1)
scores=logprob.masked_fill(~mask,0.).sum(-1)
manual=torch.stack([logprob[0,:3].sum(),logprob[1,:2].sum()])
torch.testing.assert_close(scores,manual)
scores.sum().backward()
assert logits.grad[~mask].abs().sum()==0
assert not torch.allclose(scores,scores/mask.sum(-1))
print("DPO derivative/finite difference, preferred odds, token sums and response masks passed")


# Full causal-decoder preference gradient, with a genuinely frozen reference.
import copy
import math
from llm_core import TinyLM,causal_batch,response_log_probs,BOS,EOS,STOI
torch.manual_seed(101)
policy_model=TinyLM(width=8,heads=2,depth=1).double()
reference_model=copy.deepcopy(policy_model).eval()
for p in reference_model.parameters(): p.requires_grad_(False)
ref_snapshot={k:v.clone() for k,v in reference_model.state_dict().items()}
prompt=[BOS,STOI["the"],STOI["red"]]
sequences=[prompt+[STOI["fox"],EOS],prompt+[STOI["cat"],STOI["."],EOS]]
inputs,labels,response_mask=causal_batch(sequences,[len(prompt)]*2)
scores=response_log_probs(policy_model(inputs),labels,response_mask)
with torch.no_grad():
    ref_scores=response_log_probs(reference_model(inputs),labels,response_mask)
delta=(scores[0]-ref_scores[0])-(scores[1]-ref_scores[1])
objective=F.softplus(-beta*delta)
assert math.isclose(float(objective.detach()),math.log(2))
params=list(policy_model.parameters())
actual=torch.autograd.grad(objective,params,retain_graph=True)
margin_grad=torch.autograd.grad(scores[0]-scores[1],params)
factor=-beta*torch.sigmoid(-beta*delta.detach())
for got,base_grad in zip(actual,margin_grad):
    torch.testing.assert_close(got,factor*base_grad)
with torch.no_grad():
    for p,g in zip(params,actual): p.add_(g,alpha=-.001)
    after=response_log_probs(policy_model(inputs),labels,response_mask)
    new_margin=(after[0]-ref_scores[0])-(after[1]-ref_scores[1])
assert new_margin>delta.detach()
for k,v in reference_model.state_dict().items():
    torch.testing.assert_close(v,ref_snapshot[k],rtol=0,atol=0)
assert all(p.grad is None for p in reference_model.parameters())
for row in range(2):
    sx,sy,sm=causal_batch([sequences[row]],[len(prompt)])
    torch.testing.assert_close(response_log_probs(policy_model(sx),sy,sm)[0],after[row])
# Ignored -100 labels and NaN logits never enter log_softmax/gather.
bad_logits=policy_model(inputs).detach()
bad_logits[~response_mask]=float("nan")
bad_labels=labels.clone();bad_labels[~response_mask]=-100
probe=bad_logits.requires_grad_()
safe_scores=response_log_probs(probe,bad_labels,response_mask)
torch.testing.assert_close(safe_scores,after)
safe_scores.sum().backward()
assert torch.isfinite(probe.grad).all() and probe.grad[~response_mask].abs().sum()==0
try:
    response_log_probs(probe,bad_labels,torch.zeros_like(response_mask))
    raise AssertionError("empty response accepted")
except ValueError:
    pass
# Exact score-function expectation and a prompt-only baseline, no Monte Carlo noise.
z=torch.tensor([.2,-.4,.7],dtype=torch.float64,requires_grad=True)
reward=torch.tensor([1.,-.5,2.],dtype=torch.float64)
prob=z.softmax(-1)
direct=torch.autograd.grad((prob*reward).sum(),z,retain_graph=True)[0]
surrogate=(prob.detach()*(reward-.8)*z.log_softmax(-1)).sum()
with_baseline=torch.autograd.grad(surrogate,z)[0]
torch.testing.assert_close(direct,with_baseline)
print("decoder DPO gradient/frozen reference, safe ragged response sums and baseline identity passed")
