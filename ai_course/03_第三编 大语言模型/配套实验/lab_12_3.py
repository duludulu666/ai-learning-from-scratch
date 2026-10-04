"""Token-weighted accumulation and an exact resumable CPU AdamW step."""
import copy
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from llm_core import seed_all, token_nll

seed_all()
full=torch.nn.Linear(3,4).double()
micro=copy.deepcopy(full)
x=torch.randn(5,3,dtype=torch.float64)
y=torch.tensor([0,1,2,3,1])
loss=F.cross_entropy(full(x),y)
loss.backward()
for sl in [slice(0,2),slice(2,5)]:
    # Divide each SUM loss by the TOTAL logical-batch target count.
    (F.cross_entropy(micro(x[sl]),y[sl],reduction="sum")/5).backward()
for left,right in zip(full.parameters(),micro.parameters()):
    torch.testing.assert_close(left.grad,right.grad)
# Deliberately wrong mean-of-means differs when microbatch sizes differ.
wrong=copy.deepcopy(full); wrong.zero_grad(set_to_none=True)
for sl in [slice(0,2),slice(2,5)]:
    (F.cross_entropy(wrong(x[sl]),y[sl])/2).backward()
assert not torch.allclose(wrong.weight.grad,full.weight.grad)

model=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.ReLU(),
                          torch.nn.Dropout(.25),torch.nn.Linear(8,4)).double()
optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
sampler=torch.Generator().manual_seed(91)
def update(model,optimizer,sampler):
    indices=torch.randperm(len(x),generator=sampler)
    optimizer.zero_grad(set_to_none=True)
    value=F.cross_entropy(model(x[indices]),y[indices])
    value.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return value.detach(),norm,indices

update(model,optimizer,sampler)
state=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
           torch_rng=torch.get_rng_state(),sampler_rng=sampler.get_state(),step=1,
           config=dict(dtype="float64",batch_size=5,lr=.01),data_version="fixed-fixture-v1")
with tempfile.TemporaryDirectory(prefix="part3_resume_") as tmp:
    path=Path(tmp)/"checkpoint.pt"
    torch.save(state,path)
    expected_loss,_,expected_indices=update(model,optimizer,sampler)
    expected={name:value.clone() for name,value in model.state_dict().items()}
    restored=copy.deepcopy(model)
    restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=.01)
    loaded=torch.load(path,weights_only=True)
    restored.load_state_dict(loaded["model"])
    restored_optimizer.load_state_dict(loaded["optimizer"])
    restored_sampler=torch.Generator()
    restored_sampler.set_state(loaded["sampler_rng"])
    torch.set_rng_state(loaded["torch_rng"])
    actual_loss,norm,actual_indices=update(restored,restored_optimizer,restored_sampler)
    torch.testing.assert_close(actual_loss,expected_loss,rtol=0,atol=0)
    assert torch.equal(actual_indices,expected_indices)
    for name,value in restored.state_dict().items():
        torch.testing.assert_close(value,expected[name],rtol=0,atol=0)
    assert torch.isfinite(norm)
print("weighted microbatch gradients and dropout/AdamW next-step replay passed")
print("AMP CUDA branch is documentation only in this CPU experiment.")


# Masked sequence accumulation: gradients and the actual optimizer step agree.
from llm_core import token_nll
torch.manual_seed(81)
features=torch.randn(2,5,3,dtype=torch.float64)
labels=torch.tensor([[0,1,-100,-100,-100],[1,2,3,0,1]])
mask=labels.ne(-100)
whole=torch.nn.Linear(3,4).double()
parts=copy.deepcopy(whole)
ow=torch.optim.AdamW(whole.parameters(),lr=.01)
op=torch.optim.AdamW(parts.parameters(),lr=.01)
total,count=token_nll(whole(features),labels,mask)
(total/count).backward()
for row in range(2):
    local,_=token_nll(parts(features[row:row+1]),labels[row:row+1],mask[row:row+1])
    (local/count).backward()
for p,q in zip(whole.parameters(),parts.parameters()):
    torch.testing.assert_close(p.grad,q.grad)
ow.step();op.step()
for p,q in zip(whole.parameters(),parts.parameters()):
    torch.testing.assert_close(p,q)
# Simulate average reduction for two ranks.
rank_grads=[]
for row in range(2):
    replica=copy.deepcopy(whole);replica.zero_grad(set_to_none=True)
    subtotal,_=token_nll(replica(features[row:row+1]),labels[row:row+1],mask[row:row+1])
    (2*subtotal/count).backward()
    rank_grads.append([p.grad.clone() for p in replica.parameters()])
whole.zero_grad(set_to_none=True)
sub,_=token_nll(whole(features),labels,mask)
(sub/count).backward()
for i,p in enumerate(whole.parameters()):
    torch.testing.assert_close(p.grad,(rank_grads[0][i]+rank_grads[1][i])/2)
g,scale,threshold=.01,1000.,1.
assert abs(min(scale*g,threshold)/scale-.001)<1e-12
assert min((scale*g)/scale,threshold)==g
# The earlier replay should also restore optimizer moments, not only parameters.
for p,q in zip(model.parameters(),restored.parameters()):
    for key in ["step","exp_avg","exp_avg_sq"]:
        torch.testing.assert_close(optimizer.state[p][key],restored_optimizer.state[q][key],rtol=0,atol=0)
fresh=copy.deepcopy(model);fresh.load_state_dict(loaded["model"])
fresh_opt=torch.optim.AdamW(fresh.parameters(),lr=.01)
fresh_sampler=torch.Generator();fresh_sampler.set_state(loaded["sampler_rng"])
torch.set_rng_state(loaded["torch_rng"])
update(fresh,fresh_opt,fresh_sampler)
assert any(not torch.equal(p,q) for p,q in zip(fresh.parameters(),restored.parameters()))
print("masked accumulation/AdamW, simulated rank averaging, scale/clip order and moment replay passed")
