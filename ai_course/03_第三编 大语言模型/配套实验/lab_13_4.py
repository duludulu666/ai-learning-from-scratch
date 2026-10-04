"""Decoding filters, deterministic random streams, EOS and a beam-search counterexample."""
import math
import torch
from llm_core import figure_path

def filtered_distribution(logits,temperature=1.,top_k=None,top_p=None):
    if logits.ndim!=1 or logits.numel()==0 or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("expected a finite logits vector")
    if isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or temperature<=0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive; implement greedy separately")
    # Center BEFORE dividing; float64 avoids overflow for small finite temperatures.
    centered=logits.double()-logits.double().max()
    scores=centered/temperature
    if top_k is not None:
        if type(top_k) is not int or not 1<=top_k<=len(scores): raise ValueError("bad top_k")
        # Stable ties prefer the smaller original token ID.
        indices=scores.argsort(descending=True,stable=True)[:top_k]
        keep=torch.zeros_like(scores,dtype=torch.bool); keep[indices]=True
        scores=scores.masked_fill(~keep,-torch.inf)
    if top_p is not None:
        if isinstance(top_p,bool) or not isinstance(top_p,(int,float)) or not 0<top_p<=1: raise ValueError("bad top_p")
        ordered,indices=scores.sort(descending=True,stable=True)
        probs=ordered.softmax(-1)
        # Keep the token that crosses the threshold: its preceding mass is still below p.
        preceding=torch.cat([probs.new_zeros(1),probs.cumsum(-1)[:-1]])
        remove=preceding>=top_p
        if top_p==1: remove.zero_()
        ordered=ordered.masked_fill(remove,-torch.inf)
        scores=torch.full_like(scores,-torch.inf).scatter(0,indices,ordered)
    return scores.softmax(-1)

logits=torch.tensor([.5,.3,.15,.05],dtype=torch.float64).log()
p=filtered_distribution(logits,top_p=.7)
torch.testing.assert_close(p,torch.tensor([.625,.375,0,0],dtype=torch.float64))
assert filtered_distribution(logits,top_k=1).argmax()==logits.argmax()
assert (filtered_distribution(logits,top_p=.1)>0).sum()==1
torch.testing.assert_close(filtered_distribution(logits,top_p=1.),logits.softmax(-1))
for kwargs in [dict(temperature=0),dict(top_p=0),dict(top_k=0)]:
    try:
        filtered_distribution(logits,**kwargs)
        raise AssertionError("invalid configuration accepted")
    except ValueError:
        pass
a=torch.Generator().manual_seed(7); b=torch.Generator().manual_seed(7)
draws=torch.multinomial(p,1000,replacement=True,generator=a)
assert torch.equal(draws,torch.multinomial(p,1000,replacement=True,generator=b))
assert (draws<2).all()

# Conditional tree of exactly two non-EOS tokens followed by EOS.
tree={
    (): {"A":.6,"B":.4},
    ("A",): {"x":.5,"y":.5},
    ("B",): {"x":.99,"y":.01},
}
def beam_search(width):
    if type(width) is not int or width<1: raise ValueError("beam width must be positive")
    beams=[((),0.)]
    for _ in range(2):
        candidates=[]
        for prefix,score in beams:
            for token,prob in tree[prefix].items():
                candidates.append((prefix+(token,),score+math.log(prob)))
        beams=sorted(candidates,key=lambda item:(-item[1],item[0]))[:width]
    return beams[0]
greedy=beam_search(1); wider=beam_search(2)
assert greedy[0][0]=="A" and wider[0]==("B","x")
assert math.isclose(math.exp(greedy[1]),.3)
assert math.isclose(math.exp(wider[1]),.396)
# Explicit stop policy fixture.
sequence=[]
for token in [1,2,0,3,4]:
    sequence.append(token)
    if token==0: break
assert sequence==[1,2,0]
print("greedy / width-2 beam:",greedy,wider)
print("top-p probabilities:",p.tolist())
print("filter boundary, seeded sampling, invalid arguments and EOS stop checks passed")
path=figure_path("13_4_temperature.png")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,4))
for temperature in [.3,1.,2.]:
    probs=filtered_distribution(logits,temperature)
    ax.plot(range(4),probs.numpy(),marker="o",label=f"T = {temperature}")
ax.set(xticks=range(4),xlabel="Token ID",ylabel="Probability",
       title="Temperature rescales fixed model logits")
ax.legend();fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


# Extreme temperatures, ties, endpoint semantics, and filter-combination order.
extreme=filtered_distribution(torch.tensor([10000.,9999.]),temperature=1e-300)
torch.testing.assert_close(extreme,torch.tensor([1.,0.],dtype=torch.float64))
assert torch.isfinite(extreme).all()
ties=filtered_distribution(torch.zeros(4),top_k=2)
torch.testing.assert_close(ties,torch.tensor([.5,.5,0.,0.],dtype=torch.float64))
for value in [torch.empty(0),torch.tensor([1,2]),torch.tensor([float("nan")])]:
    try:
        filtered_distribution(value)
        raise AssertionError("bad logits accepted")
    except ValueError:
        pass
for kwargs in [dict(temperature=True),dict(top_p=True),dict(top_p=float("nan"))]:
    try:
        filtered_distribution(logits,**kwargs)
        raise AssertionError("bad filter argument accepted")
    except ValueError:
        pass
combo=torch.tensor([.5,.25,.2,.05],dtype=torch.float64).log()
assert (filtered_distribution(combo,top_p=.8)>0).sum()==3
assert (filtered_distribution(combo,top_k=2,top_p=.8)>0).sum()==2
# Correct-candidate availability can rise while a bad selector always picks a failure.
candidate_correct=torch.tensor([[1,0,0],[0,1,0],[0,0,1]],dtype=torch.bool)
bad_verifier=(~candidate_correct).double()
selected=bad_verifier.argmax(-1)
assert candidate_correct.any(-1).all()
assert not candidate_correct[torch.arange(3),selected].any()
print("tiny-temperature stability, tie policy, exact endpoints, filter order and bad-verifier counterexample passed")
