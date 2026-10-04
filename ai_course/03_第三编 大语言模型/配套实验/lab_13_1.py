"""Response-only SFT of a genuinely pretrained tiny decoder; no downloaded model."""
import copy
import itertools
import math
import random
import torch
from llm_core import (train_base, evaluate, fit_model, causal_batch, token_nll,
                      COLORS,ANIMALS,BOS,EOS,USER,ASSISTANT,STOI,decode,figure_path)

model, base_splits, _ = train_base()
base_snapshot=copy.deepcopy(model.state_dict())
base_before=evaluate(model,base_splits[1])[0]
pairs=list(itertools.product(COLORS,ANIMALS))
random.Random(47).shuffle(pairs)
def example(pair):
    color,animal=pair
    prefix=[BOS,USER,STOI["describe"],STOI[color],STOI[animal],ASSISTANT]
    response=[STOI["the"],STOI[color],STOI[animal],STOI["."],EOS]
    return prefix+response,len(prefix),prefix,response
examples=[example(pair) for pair in pairs]
train_examples,val_examples,test_examples=examples[:10],examples[10:13],examples[13:]
def unpack(items):
    return [item[0] for item in items],[item[1] for item in items]
train,starts=unpack(train_examples)
val,val_starts=unpack(val_examples)
test,test_starts=unpack(test_examples)
before=evaluate(model,val,val_starts)[0]
# Masked logits receive zero direct loss gradient; prompt embeddings may still receive gradients.
x,y,mask=causal_batch(train[:2],starts[:2])
model.zero_grad(set_to_none=True)
logits=model(x); logits.retain_grad()
total,count=token_nll(logits,y,mask)
(total/count).backward()
assert torch.equal(logits.grad[~mask],torch.zeros_like(logits.grad[~mask]))
assert model.token.weight.grad[USER].norm()>0
assert mask.sum().item()==10
records=fit_model(model,train,val,steps=450,lr=.002,starts=starts,
                  validation_starts=val_starts,seed=53,batch_size=16)
after=evaluate(model,val,val_starts)[0]
test_loss,test_accuracy=evaluate(model,test,test_starts)
base_after=evaluate(model,base_splits[1])[0]
exact=0
for _,_,prefix,response in test_examples:
    generated=model.generate(prefix,max_new=len(response))
    prediction=generated[len(prefix):]
    exact+=int(prediction==response)
    print("prompt:",decode(prefix),"response:",decode(prediction),"expected:",decode(response))
assert after<before
assert any(not torch.equal(value,base_snapshot[name]) for name,value in model.state_dict().items())
print("SFT validation before/after:",before,after)
print("SFT test NLL/token accuracy/free exact:",test_loss,test_accuracy,exact/len(test_examples))
print("base grammar validation before/after SFT:",base_before,base_after)
print("response shift, direct-logit mask, contextual prompt gradient and parameter updates passed")
path=figure_path("13_1_sft.png")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(*zip(*records),marker=".")
ax.set(xlabel="SFT optimizer updates",ylabel="Response-token validation NLL",
       title="Response-only adaptation on a tiny held-out split")
fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


# Ragged responses: selected loss/gradients equal separate-example sums.
ragged=[train[0],train[1][:-3]+[EOS]]
ragged_starts=starts[:2]
bx,by,bm=causal_batch(ragged,ragged_starts)
assert bm.sum(1).tolist()==[5,3]
model.zero_grad(set_to_none=True)
combined,count=token_nll(model(bx),by,bm)
(combined/count).backward()
batch_grads=[p.grad.clone() for p in model.parameters()]
model.zero_grad(set_to_none=True)
separate_sum=0.
for sequence,start in zip(ragged,ragged_starts):
    sx,sy,sm=causal_batch([sequence],[start])
    subtotal,_=token_nll(model(sx),sy,sm)
    separate_sum+=float(subtotal.detach())
    (subtotal/count).backward()
assert math.isclose(float(combined.detach()),separate_sum,rel_tol=1e-5)
for p,g in zip(model.parameters(),batch_grads):
    torch.testing.assert_close(p.grad,g,atol=2e-6,rtol=2e-5)
# Two assistant spans; role delimiters remain context-only in this fixture.
tokens=[BOS,USER,STOI["describe"],ASSISTANT,STOI["red"],EOS,
        USER,STOI["describe"],ASSISTANT,STOI["blue"],EOS]
original_flags=torch.tensor([0,0,0,0,1,1,0,0,0,1,1],dtype=torch.bool)
shifted_flags=original_flags[1:]
assert shifted_flags.nonzero().flatten().tolist()==[3,4,8,9]
assert not shifted_flags[5:8].any()
assert set(STOI[word] for word in ["describe"]) | {USER,ASSISTANT} <= set(range(model.head.out_features))
for sequences,starts_bad in [([],None),([[BOS,EOS]],[2]),([[BOS,EOS]],[True])]:
    try:
        causal_batch(sequences,starts_bad)
        raise AssertionError("invalid supervised example accepted")
    except ValueError:
        pass
print("ragged response loss/parameter gradients, multi-turn target mask and empty-response rejection passed")
