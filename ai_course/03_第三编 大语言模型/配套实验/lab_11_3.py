"""Train an independent tiny causal LM on a finite artificial grammar."""
import math
import torch
from llm_core import (train_base,evaluate,causal_batch,token_nll,decode,
                      BOS,EOS,VOCAB,STOI,figure_path)

model, (train,validation,test), records = train_base()
test_nll, test_accuracy = evaluate(model,test)
xtrain,ytrain,valid = causal_batch(train)
counts = torch.bincount(ytrain[valid],minlength=VOCAB).double()+1
unigram = counts/counts.sum()
_, ytest, test_valid = causal_batch(test)
baseline = float(-unigram[ytest[test_valid]].log().mean())
model.eval()
x,_,_=causal_batch(test[:2])
with torch.no_grad():
    logits = model(x)
    perturbed = x.clone()
    perturbed[:,5:] = (perturbed[:,5:]+1)%VOCAB
    torch.testing.assert_close(logits[:,:5],model(perturbed)[:,:5])
    padded = torch.cat([x,torch.zeros((2,3),dtype=torch.long)],1)
    torch.testing.assert_close(logits,model(padded)[:,:x.shape[1]])
assert test_nll < baseline
print("parameters:",sum(p.numel() for p in model.parameters()))
print("first/best validation NLL:",records[0][1],min(v for _,v in records))
print("test NLL / PPL / accuracy:",test_nll,math.exp(test_nll),test_accuracy)
print("train-fitted unigram test NLL:",baseline)
print("population grammar entropy per predicted token:",math.log(768)/9)
for color in ["red","blue"]:
    prefix = [BOS,STOI["the"],STOI[color]]
    generated=model.generate(prefix,max_new=10)
    print("greedy sample:",decode(generated))
for prefix,max_new in [([],1),([BOS],-1),([BOS],True),([VOCAB],1),([BOS]*24,2)]:
    try:
        model.generate(prefix,max_new)
        raise AssertionError("invalid generation request accepted")
    except ValueError:
        pass
assert model.generate([BOS],0)==[BOS]
print("disjoint splits, future perturbation, right-padding and generation-boundary checks passed")
path=figure_path("11_3_lm_curve.png")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(*zip(*records),marker=".",label="Validation NLL")
ax.axhline(baseline,ls="--",label="Unigram TEST NLL (reference)")
ax.axhline(math.log(768)/9,ls=":",label="Population grammar entropy")
ax.set(xlabel="Optimizer updates",ylabel="Nats / predicted token",title="Tiny grammar LM")
ax.legend()
fig.tight_layout()
fig.savefig(path,dpi=150)
plt.close(fig)


# Exact parameter ledger and the uniform-grammar accuracy reference.
width=32;vocab=VOCAB;depth=2;context=24
per_block=8*width**2+11*width  # QKV/out, two FFN maps, and two LayerNorms.
expected_count=2*vocab*width+context*width+depth*per_block+2*width
assert expected_count==19136==sum(p.numel() for p in model.parameters())
oracle_accuracy=(4+4/4+1/3)/9
assert math.isclose(oracle_accuracy,16/27)
print("population Bayes token accuracy:",oracle_accuracy)
# Evaluation and generation preserve the caller's top-level mode and model weights.
model.train()
snapshot={name:value.clone() for name,value in model.state_dict().items()}
evaluate(model,validation)
assert model.training
assert model.generate([BOS,EOS],3)==[BOS,EOS]
assert model.training
for name,value in model.state_dict().items():
    torch.testing.assert_close(value,snapshot[name],rtol=0,atol=0)
print("parameter ledger, oracle accuracy, terminal prefix and evaluation-state checks passed")


with torch.no_grad():
    tx,ty,tm=causal_batch(test)
    position_accuracy=(model(tx).argmax(-1)==ty).double().mean(0)
print("test accuracy by target position (the,color,animal,verb,the,color,animal,.,EOS):",
      position_accuracy.tolist())
