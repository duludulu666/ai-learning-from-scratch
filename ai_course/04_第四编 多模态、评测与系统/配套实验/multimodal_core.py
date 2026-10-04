"""Small CPU teaching components, not pretrained or production VLMs."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

def setup(seed=15):
    torch.manual_seed(seed)
    torch.set_num_threads(1)

def bars(n, seed):
    """Independent noisy images; label 0 horizontal, 1 vertical."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    x = 0.08 * torch.randn(n, 1, 8, 8, generator=g)
    offsets = torch.randint(1, 7, (n,), generator=g)
    for i, (label, offset) in enumerate(zip(y, offsets)):
        if label == 0:
            x[i, 0, offset, :] += 1
        else:
            x[i, 0, :, offset] += 1
    return x, y

class TinyViT(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.patch = nn.Conv2d(1, width, kernel_size=2, stride=2)
        self.pos = nn.Parameter(torch.randn(1, 16, width) * 0.02)
        block = nn.TransformerEncoderLayer(width, 2, 32, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)

    def tokens(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 8, 8) or x.shape[0] == 0:
            raise ValueError("TinyViT expects a nonempty (B,1,8,8) image batch")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Images must be finite floating-point tensors")
        z = self.patch(x).flatten(2).transpose(1, 2)
        return self.norm(self.encoder(z + self.pos))

    def forward(self, x):
        return self.head(self.tokens(x).mean(1))

class TinyVLM(nn.Module):
    """Causal text self-attention followed by image cross-attention.
    Vocabulary: PAD=0, BOS=1, QUESTION=2, HORIZONTAL=3, VERTICAL=4, EOS=5.
    """
    def __init__(self):
        super().__init__()
        self.vision = TinyViT()
        self.vision.head = nn.Identity()  # unused classification head removed
        self.embed = nn.Embedding(6, 16)
        self.pos = nn.Parameter(torch.randn(1, 5, 16) * 0.02)
        self.self_attn = nn.MultiheadAttention(16, 2, dropout=0, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(16, 2, dropout=0, batch_first=True)
        self.ln1, self.ln2, self.ln3 = [nn.LayerNorm(16) for _ in range(3)]
        self.ff = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
        self.out = nn.Linear(16, 6)

    def forward(self, image, ids):
        # Fixed-length, unpadded teaching batches; PAD is not a supported input token.
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.pos.shape[1]:
            raise ValueError("Text must have shape (B,T) with 1 <= T <= 5")
        if ids.dtype != torch.long or not ((ids > 0) & (ids < 6)).all():
            raise ValueError("Text IDs must be int64, in 1..5, without PAD")
        if image.shape[0] != ids.shape[0]:
            raise ValueError("Image and text batches must match")
        visual = self.vision.tokens(image)
        h = self.embed(ids) + self.pos[:, :ids.shape[1]]
        blocked = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device).triu(1)
        q = self.ln1(h)
        h = h + self.self_attn(q, q, q, attn_mask=blocked, need_weights=False)[0]
        h = h + self.cross_attn(self.ln2(h), visual, visual, need_weights=False)[0]
        h = h + self.ff(self.ln3(h))
        return self.out(h)

def answer_loss(logits, targets):
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or targets.dtype != torch.long:
        raise ValueError("Need logits (B,T,V) and int64 targets (B,T)")
    active = targets != -100
    if not active.any():
        raise ValueError("No supervised response tokens")
    selected = targets[active]
    if not ((selected >= 0) & (selected < logits.shape[-1])).all():
        raise ValueError("Active targets must be vocabulary IDs")
    if not torch.isfinite(logits[active]).all():
        raise ValueError("Active logits must be finite")
    return F.cross_entropy(logits[active], selected)

def save_figure(fig, name):
    path = Path(__file__).resolve().parent.parent / "图示" / name
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("figure:", path.name)
