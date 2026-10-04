"""Small, CPU-first language-model components for Part III; no downloads."""
from pathlib import Path
import copy
import itertools
import math
import random

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PAD, BOS, EOS, USER, ASSISTANT = range(5)
COLORS = ("red", "blue", "green", "gold")
ANIMALS = ("fox", "owl", "cat", "dog")
VERBS = ("sees", "follows", "helps")
WORDS = ("<pad>", "<bos>", "<eos>", "<user>", "<assistant>",
         "the", ".", "describe") + COLORS + ANIMALS + VERBS
STOI = {word: index for index, word in enumerate(WORDS)}
VOCAB = len(WORDS)

def seed_all(seed=17):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

def encode(words):
    return [STOI[word] for word in words.split()]

def decode(ids):
    return " ".join(WORDS[int(index)] for index in ids)

class ByteBPE:
    """Educational byte BPE: no normalization or production pre-tokenizer."""
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.fitted = False

    @staticmethod
    def replace(sequence, pair, new_id):
        result, index = [], 0
        while index < len(sequence):
            if index + 1 < len(sequence) and tuple(sequence[index:index+2]) == pair:
                result.append(new_id)
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return result

    def fit(self, documents, num_merges=20):
        if self.fitted:
            raise ValueError("fit a fresh tokenizer, not a previously fitted one")
        if type(num_merges) is not int or num_merges < 0:
            raise ValueError("num_merges must be a nonnegative integer")
        if isinstance(documents, str):
            raise ValueError("provide a collection of documents, not one bare string")
        documents = list(documents)
        if any(not isinstance(document, str) for document in documents):
            raise ValueError("each document must be a string")
        sequences = [list(document.encode("utf-8")) for document in documents]
        self.fitted = True
        for _ in range(num_merges):
            counts = {}
            for sequence in sequences:
                for pair in zip(sequence, sequence[1:]):
                    counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            # Highest frequency first, deterministic tie-breaking by token IDs.
            pair = min(counts, key=lambda item: (-counts[item], item))
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges.append((pair, new_id))
            sequences = [self.replace(sequence, pair, new_id) for sequence in sequences]
        return self

    def encode(self, text):
        sequence = list(text.encode("utf-8"))
        for pair, new_id in self.merges:
            sequence = self.replace(sequence, pair, new_id)
        return sequence

    def decode(self, ids):
        # Join bytes before decoding: an individual byte token need not be valid UTF-8.
        return b"".join(self.vocab[index] for index in ids).decode("utf-8")

def token_nll(logits, targets, valid):
    """Return SUM NLL and valid count. Select before CE to exclude invalid labels."""
    if logits.ndim < 2 or logits.shape[-1] < 1 or not logits.is_floating_point():
        raise ValueError("logits need a nonempty vocabulary axis and floating dtype")
    if targets.dtype != torch.long:
        raise ValueError("targets must be int64 token IDs")
    if logits.device != targets.device or targets.device != valid.device:
        raise ValueError("logits, targets and mask must share a device")
    if logits.shape[:-1] != targets.shape or valid.shape != targets.shape:
        raise ValueError("incompatible logits/targets/mask shapes")
    if valid.dtype != torch.bool or not valid.any():
        raise ValueError("need a boolean mask with at least one valid target")
    selected_logits, selected_targets = logits[valid], targets[valid]
    if not torch.isfinite(selected_logits).all():
        raise ValueError("valid logits must be finite")
    if ((selected_targets < 0) | (selected_targets >= logits.shape[-1])).any():
        raise ValueError("a supervised target is outside the vocabulary")
    return F.cross_entropy(selected_logits, selected_targets, reduction="sum"), int(valid.sum())

def response_log_probs(logits, targets, valid):
    """Sum valid token log probabilities per response; select before log_softmax/gather."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("expected (batch,time,vocab) logits and (batch,time) targets")
    token_nll(logits, targets, valid)  # Common shape/dtype/value validation.
    if not valid.any(dim=1).all():
        raise ValueError("each preference response needs at least one scored target")
    selected = logits[valid].log_softmax(-1)
    token_scores = selected.gather(-1, targets[valid].unsqueeze(-1)).squeeze(-1)
    row_ids = torch.arange(targets.shape[0], device=targets.device)[:, None].expand_as(targets)[valid]
    return logits.new_zeros(targets.shape[0]).index_add(0, row_ids, token_scores)

def causal_batch(sequences, target_starts=None):
    """target_starts stores the first supervised index in each ORIGINAL sequence."""
    if not sequences or any(len(sequence) < 2 for sequence in sequences):
        raise ValueError("need nonempty sequences with at least two tokens")
    if any(type(token) is not int or not 0 <= token < VOCAB for sequence in sequences for token in sequence):
        raise ValueError("sequence tokens must be vocabulary integer IDs")
    length = max(map(len, sequences))
    ids = torch.full((len(sequences), length), PAD, dtype=torch.long)
    valid = torch.zeros((len(sequences), length-1), dtype=torch.bool)
    if target_starts is None:
        target_starts = [1] * len(sequences)
    if len(target_starts) != len(sequences):
        raise ValueError("one start index per sequence")
    for row, (sequence, start) in enumerate(zip(sequences, target_starts)):
        if type(start) is not int or not 1 <= start < len(sequence):
            raise ValueError("invalid sequence length or supervision start")
        ids[row, :len(sequence)] = torch.tensor(sequence)
        valid[row, start-1:len(sequence)-1] = True
    return ids[:, :-1], ids[:, 1:], valid

class CausalAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads, self.dim = heads, width // heads
        self.qkv = nn.Linear(width, 3*width)
        self.out = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        def split(tensor):
            return tensor.reshape(batch, length, self.heads, self.dim).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
        allowed = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        return self.out((weights @ v).transpose(1, 2).reshape(batch, length, width))

class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = CausalAttention(width, heads)
        self.ffn = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

class TinyLM(nn.Module):
    """Same pre-LN building blocks as Part II; independent local module for portability."""
    def __init__(self, width=32, heads=4, depth=2, max_length=24):
        super().__init__()
        self.config = dict(width=width, heads=heads, depth=depth, max_length=max_length)
        self.max_length = max_length
        self.token = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, VOCAB, bias=False)

    def forward(self, ids):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.max_length:
            raise ValueError("expected (batch, length) within context limit")
        x = self.token(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))[None]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, prefix, max_new=12):
        if type(max_new) is not int or max_new < 0:
            raise ValueError("max_new must be a nonnegative integer")
        if not prefix or len(prefix) > self.max_length or len(prefix) + max_new - 1 > self.max_length:
            raise ValueError("generation request exceeds context capacity or has empty prefix")
        if any(type(token) is not int or not 0 <= token < VOCAB for token in prefix):
            raise ValueError("prefix token IDs must be vocabulary integers")
        old_mode = self.training
        self.eval()
        result = list(prefix)
        try:
            if result[-1] == EOS:
                return result
            for _ in range(max_new):
                next_id = int(self(torch.tensor([result], device=self.token.weight.device))[0, -1].argmax())
                result.append(next_id)
                if next_id == EOS:
                    break
        finally:
            self.train(old_mode)
        return result

def grammar_splits():
    sequences = []
    for c1, a1, verb, c2, a2 in itertools.product(COLORS, ANIMALS, VERBS, COLORS, ANIMALS):
        sentence = f"the {c1} {a1} {verb} the {c2} {a2} ."
        sequences.append([BOS] + encode(sentence) + [EOS])
    random.Random(23).shuffle(sequences)
    train, validation, test = sequences[:384], sequences[384:480], sequences[480:576]
    groups = [set(map(tuple, group)) for group in (train, validation, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    return train, validation, test

@torch.no_grad()
def evaluate(model, sequences, starts=None):
    old_mode = model.training
    model.eval()
    try:
        x, y, mask = causal_batch(sequences, starts)
        logits = model(x)
        nll, count = token_nll(logits, y, mask)
        accuracy = float((logits.argmax(-1)[mask] == y[mask]).float().mean())
        return float(nll / count), accuracy
    finally:
        model.train(old_mode)

def fit_model(model, train, validation, steps=300, lr=0.003, starts=None,
              validation_starts=None, seed=29, batch_size=32):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    first = evaluate(model, validation, validation_starts)[0]
    records, best_loss, best_state = [(0, first)], first, copy.deepcopy(model.state_dict())
    for step in range(1, steps+1):
        indices = torch.randint(len(train), (batch_size,), generator=generator).tolist()
        selected = [train[index] for index in indices]
        selected_starts = None if starts is None else [starts[index] for index in indices]
        x, y, mask = causal_batch(selected, selected_starts)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count = token_nll(model(x), y, mask)
        (total / count).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == steps:
            score = evaluate(model, validation, validation_starts)[0]
            records.append((step, score))
            if score < best_loss:
                best_loss, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return records

def train_base(steps=300):
    seed_all(17)
    train, validation, test = grammar_splits()
    model = TinyLM()
    records = fit_model(model, train, validation, steps=steps)
    return model, (train, validation, test), records

def figure_path(name):
    import matplotlib
    matplotlib.use("Agg")
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    return folder / name
