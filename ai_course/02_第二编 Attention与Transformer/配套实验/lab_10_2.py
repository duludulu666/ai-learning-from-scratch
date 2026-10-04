"""Offset-correct KV caching and exact blockwise softmax verification."""

import torch
from transformer_core import TinyTransformer
from cache_core import cached_forward

torch.manual_seed(34)
torch.set_num_threads(1)
model = TinyTransformer(max_length=20).double().eval()
tokens = torch.randint(1, 9, (2, 8))
with torch.no_grad():
    full = model(tokens)
cache, outputs, offset = None, [], 0
for chunk_length in [3, 1, 4]:
    chunk = tokens[:, offset:offset+chunk_length]
    logits, cache = cached_forward(model, chunk, cache)
    outputs.append(logits)
    offset += chunk_length
streamed = torch.cat(outputs, dim=1)
torch.testing.assert_close(streamed, full, atol=1e-9, rtol=1e-7)

prompt = tokens[:, :4]
uncached = model.generate(prompt, new_tokens=4)
prefix_logits, cache = cached_forward(model, prompt)
generated = prompt.clone()
for step in range(4):
    next_id = prefix_logits[:, -1].argmax(-1, keepdim=True)
    generated = torch.cat([generated, next_id], dim=1)
    if step < 3:
        prefix_logits, cache = cached_forward(model, next_id, cache)
torch.testing.assert_close(generated, uncached)
print("chunked logits maximum difference:", (streamed - full).abs().max().item())
print("cached and uncached generation match")


# All partitions must represent the same causal sequence.
for chunks in [[1] * 8, [8], [2, 2, 2, 2]]:
    cache, pieces, offset = None, [], 0
    for length in chunks:
        logits, cache = cached_forward(model, tokens[:, offset:offset+length], cache)
        pieces.append(logits)
        offset += length
    torch.testing.assert_close(torch.cat(pieces, 1), full, atol=1e-9, rtol=1e-7)

# Invalid cache metadata should fail loudly, before a misleading broadcast.
_, valid_cache = cached_forward(model, tokens[:, :3])
def expect_invalid(candidate, message):
    try:
        cached_forward(model, tokens[:, 3:4], candidate)
    except ValueError:
        return
    raise AssertionError("invalid cache accepted: " + message)

expect_invalid(valid_cache[:-1], "missing layer")
expect_invalid([None] * len(model.blocks), "incomplete pair")
for label, transform in [
    ("value batch", lambda k, v: (k, v[:1])),
    ("heads", lambda k, v: (k[:, :1], v[:, :1])),
    ("width", lambda k, v: (k[..., :1], v[..., :1])),
    ("length", lambda k, v: (k[:, :, :-1], v[:, :, :-1])),
    ("dtype", lambda k, v: (k.float(), v.float())),
    ("rank", lambda k, v: (k[0], v[0])),
]:
    invalid = list(valid_cache)
    invalid[-1] = transform(*invalid[-1])
    expect_invalid(invalid, label)

try:
    cached_forward(model, torch.ones(2, 21, dtype=torch.long))
except ValueError:
    pass
else:
    raise AssertionError("context overflow accepted")

model.train()
try:
    cached_forward(model, tokens[:, :1])
except ValueError:
    pass
else:
    raise AssertionError("training cache accepted")
model.eval()
print("token/chunk partitions and invalid cache cases verified")

torch.manual_seed(36)
scores = 1000 + torch.randn(17, dtype=torch.float64)
values = torch.randn(17, 5, dtype=torch.float64)
running_max = torch.tensor(-torch.inf, dtype=torch.float64)
denominator = torch.tensor(0.0, dtype=torch.float64)
numerator = torch.zeros(5, dtype=torch.float64)

for start in range(0, len(scores), 4):
    block = scores[start:start+4]
    block_values = values[start:start+4]
    new_max = torch.maximum(running_max, block.max())
    old_scale = (running_max - new_max).exp()
    exponentials = (block - new_max).exp()
    denominator = old_scale * denominator + exponentials.sum()
    numerator = old_scale * numerator + exponentials @ block_values
    running_max = new_max

streaming_read = numerator / denominator
dense_read = scores.softmax(0) @ values
torch.testing.assert_close(streaming_read, dense_read, atol=1e-12, rtol=1e-10)
print("stable blockwise softmax read matches dense softmax")
