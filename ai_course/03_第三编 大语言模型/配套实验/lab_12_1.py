"""Controlled data hygiene: metadata, exact/near duplicates, packing and mixtures."""
import hashlib
import re
import torch

def normalize(text):
    return " ".join(text.lower().split())

def fingerprint(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()

def deduplicate(records):
    seen, result = set(), []
    for record in records:
        key = fingerprint(record["text"])
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result

records = [
    dict(id="a1", source="site-a", license="permitted", text="The red fox."),
    dict(id="a2", source="mirror", license="permitted", text="  The RED fox.  "),
    dict(id="b1", source="site-b", license="unknown", text="A blue owl."),
    dict(id="c1", source="site-c", license="permitted", text="The red fox jumps."),
]
usable = [r for r in records if r["license"]=="permitted"]
unique = deduplicate(usable)
assert [r["id"] for r in unique] == ["a1","c1"]
benchmark = ["the red fox."]
clean = [r for r in unique if fingerprint(r["text"]) not in set(map(fingerprint,benchmark))]
assert [r["id"] for r in clean] == ["c1"]
def jaccard(left,right):
    a,b=set(normalize(left).split()),set(normalize(right).split())
    return len(a&b)/len(a|b) if a|b else 1.
assert fingerprint("the red fox") != fingerprint("the red fox jumps")
assert jaccard("the red fox","the red fox jumps")==.75
# A fixed threshold is a teaching choice, not a universal deduplication rule.
print("raw/permitted/exact-unique/benchmark-clean:",len(records),len(usable),len(unique),len(clean))
print("near-duplicate token Jaccard:",jaccard("the red fox","the red fox jumps"))
# Packed independent documents: block-diagonal causal visibility and no cross-doc target.
segments=torch.tensor([0,0,0,1,1,1])
index=torch.arange(len(segments))
allowed=(segments[:,None]==segments[None,:]) & (index[:,None]>=index[None,:])
assert not allowed[3:,:3].any()
valid_transition=segments[:-1].eq(segments[1:])
assert valid_transition.tolist()==[True,True,False,True,True]
# EOS alone would not prevent a later document from attending an earlier one.
plain_causal=index[:,None]>=index[None,:]
assert plain_causal[4,1] and not allowed[4,1]
sizes=torch.tensor([900.,100.],dtype=torch.float64)
for alpha in [1.,.5,0.]:
    weights=sizes.pow(alpha); weights/=weights.sum()
    print("mixture alpha/probabilities:",alpha,weights.tolist())
print("provenance gate, dedup, contamination fixture, packing and mixture checks passed")


# Actual read isolation, not only an assertion about mask entries.
q=torch.zeros(6,2,dtype=torch.float64)
k=torch.zeros_like(q)
v=torch.arange(12,dtype=torch.float64).reshape(6,2)
def read_with(mask,values):
    scores=q@k.T
    return scores.masked_fill(~mask,-torch.inf).softmax(-1)@values
changed=v.clone();changed[:3]+=100
torch.testing.assert_close(read_with(allowed,v)[3:],read_with(allowed,changed)[3:])
assert not torch.allclose(read_with(plain_causal,v)[3:],read_with(plain_causal,changed)[3:])
# Document mixture weights need not equal resulting token fractions.
doc_prob=torch.tensor([.5,.5])
lengths=torch.tensor([10.,100.])
token_share=doc_prob*lengths/(doc_prob*lengths).sum()
torch.testing.assert_close(token_share,torch.tensor([1/11,10/11]))
print("packed read isolation/negative control and document-vs-token weighting passed")
