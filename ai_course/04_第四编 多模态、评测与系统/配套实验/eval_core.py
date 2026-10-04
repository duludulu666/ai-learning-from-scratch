"""Small CPU reference metrics. Ranking scores need not be probabilities."""
import math
from numbers import Integral, Real
import torch

def _inputs(y, values, *, probabilities=True):
    y = torch.as_tensor(y, device="cpu")
    values = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if y.ndim != 1 or values.ndim != 1 or y.shape != values.shape or not y.numel():
        raise ValueError("Expected equal, nonempty one-dimensional vectors")
    if not torch.all((y == 0) | (y == 1)):
        raise ValueError("Labels must be 0 or 1")
    if not torch.isfinite(values).all():
        raise ValueError("Values must be finite")
    if probabilities and not torch.all((values >= 0) & (values <= 1)):
        raise ValueError("Probabilities must be in [0,1]")
    return y.long(), values

def _threshold(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("Threshold must be a finite real scalar")
    if not 0 <= value <= 1:
        raise ValueError("Probability threshold must be in [0,1]")

def metrics(y, p, threshold=0.5):
    y, p = _inputs(y, p)
    _threshold(threshold)
    pred = p >= threshold
    tp = ((y == 1) & pred).sum().item()
    fp = ((y == 0) & pred).sum().item()
    fn = ((y == 1) & ~pred).sum().item()
    tn = ((y == 0) & ~pred).sum().item()
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=f1, accuracy=(tp + tn) / len(y), brier=((p-y)**2).mean().item())

def auroc(y, scores):
    """Pairwise O(n_positive*n_negative) reference, not a large-data implementation."""
    y, scores = _inputs(y, scores, probabilities=False)
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return None
    diff = pos[:, None] - neg[None, :]
    return ((diff > 0).double() + 0.5 * (diff == 0).double()).mean().item()

def average_precision(y, scores):
    """Threshold-grouped AP; ties enter together, not in arbitrary row order."""
    y, scores = _inputs(y, scores, probabilities=False)
    if y.sum() == 0:
        return None
    previous_recall = 0.0
    ap = 0.0
    for cutoff in scores.unique().sort(descending=True).values:
        selected = scores >= cutoff
        recall = y[selected].sum().item() / y.sum().item()
        precision = y[selected].double().mean().item()
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap

def reliability(y, p, bins=5):
    y, p = _inputs(y, p)
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError("bins must be a positive integer")
    index = torch.clamp((p * bins).long(), max=bins-1)
    rows = []
    ece = 0.0
    for k in range(bins):
        use = index == k
        if use.any():
            conf = p[use].mean().item()
            rate = y[use].double().mean().item()
            n = use.sum().item()
            rows.append((n, conf, rate))
            ece += n / len(y) * abs(conf - rate)
    return ece, rows

def paired_bootstrap(a, b, seed=172, repeats=2000):
    """Percentile interval for mean(b-a), resampling paired independent units.
    Scores may be continuous; this computes a difference of means, not F1.
    """
    a = torch.as_tensor(a, dtype=torch.float64, device="cpu")
    b = torch.as_tensor(b, dtype=torch.float64, device="cpu")
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("Need paired nonempty one-dimensional arrays")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Paired outcomes must be finite")
    if isinstance(repeats, bool) or not isinstance(repeats, Integral) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(a), (repeats, len(a)), generator=g)
    means = (b-a)[idx].mean(1)
    return (b-a).mean().item(), torch.quantile(means, torch.tensor([.025,.975], dtype=torch.float64)).tolist()

def selective_risk(correct, confidence, threshold):
    """Risk is None when no predictions are accepted; always report coverage."""
    correct, confidence = _inputs(correct, confidence)
    _threshold(threshold)
    accept = confidence >= threshold
    coverage = accept.double().mean().item()
    risk = (1-correct[accept].double()).mean().item() if accept.any() else None
    return coverage, risk
