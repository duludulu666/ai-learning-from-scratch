"""Byte BPE: fitted on training text only; lossless Unicode round trips."""
from llm_core import ByteBPE, figure_path

train = ["low lower lowest", "low low lower", "new newer newest"] * 5
validation = ["lower newer", "你好，world!", "🦊 fox", "", "a  b\nc"]
tokenizer = ByteBPE().fit(train, num_merges=20)
snapshot = list(tokenizer.merges)
for text in train + validation:
    assert tokenizer.decode(tokenizer.encode(text)) == text
assert tokenizer.merges == snapshot
assert ByteBPE.replace([1,1,1], (1,1), 256) == [256,1]
assert len(tokenizer.vocab) == 256 + len(tokenizer.merges)
for pair, index in tokenizer.merges[:5]:
    print("merge:", pair, "->", index, repr(tokenizer.vocab[index]))
labels, before, after = [], [], []
for text in validation[:-1]:
    labels.append(repr(text))
    before.append(len(text.encode("utf-8")))
    after.append(len(tokenizer.encode(text)))
    assert after[-1] <= before[-1]
print("bytes / BPE:", list(zip(labels,before,after)))
print("round trips, deterministic nonoverlap, and frozen merges passed")
path = figure_path("11_1_tokens.png")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,4))
x = list(range(len(labels)))
ax.bar([i-.18 for i in x], before, width=.36, label="UTF-8 bytes")
ax.bar([i+.18 for i in x], after, width=.36, label="BPE tokens")
ax.set(xticks=x, xticklabels=["English", "Chinese + English", "Emoji + English", "Empty"],
       ylabel="Sequence length", title="Token count depends on text and learned merges")
ax.legend()
fig.tight_layout()
fig.savefig(path, dpi=150)
plt.close(fig)


# Independent inference algorithm: repeatedly choose the currently best-ranked pair.
import random
def ranked_encode(text, merges):
    result=list(text.encode("utf-8"))
    ranks={pair:(rank,new_id) for rank,(pair,new_id) in enumerate(merges)}
    while True:
        candidates=[ranks[pair]+(pair,) for pair in zip(result,result[1:]) if pair in ranks]
        if not candidates: return result
        _,new_id,pair=min(candidates)
        result=ByteBPE.replace(result,pair,new_id)
rng=random.Random(37)
for _ in range(100):
    text="".join(rng.choice(" lownewr你好🦊") for _ in range(rng.randrange(30)))
    assert tokenizer.encode(text)==ranked_encode(text,tokenizer.merges)
    assert tokenizer.decode(tokenizer.encode(text))==text
# Explicit ordered merges: a,b -> AB first; AB,c -> ABC second.
ordered=ByteBPE().fit(["abc"]*4,num_merges=2)
assert ordered.merges[0][0]==(97,98)
assert ordered.encode("abc")==[257]
assert ordered.decode([257])=="abc"
empty_fit=ByteBPE().fit([],0)
try:
    empty_fit.fit(["abc"],1)
    raise AssertionError("second fit accepted after zero-merge fit")
except ValueError:
    pass
for count in [-1,True,1.5]:
    try:
        ByteBPE().fit(["abc"],count)
        raise AssertionError("invalid merge count accepted")
    except ValueError:
        pass
# Byte representability does not make arbitrary sampled byte sequences valid UTF-8.
try:
    ByteBPE().decode([255])
    raise AssertionError("invalid UTF-8 sequence unexpectedly decoded")
except UnicodeDecodeError:
    pass
print("ranked inference equivalence, Unicode fuzz round trips, merge order and fit boundaries passed")
