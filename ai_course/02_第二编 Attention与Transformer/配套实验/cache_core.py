"""Inference-only, unpadded KV caching for the teaching decoder."""

import torch
import torch.nn.functional as F

@torch.no_grad()
def cached_forward(model, ids, caches=None):
    if any(module.training for module in model.modules()):
        raise ValueError("cached_forward requires every module in evaluation mode")
    if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] == 0:
        raise ValueError("provide a nonempty batch of token IDs")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token IDs must be integers")
    if ids.device != model.token.weight.device:
        raise ValueError("token IDs and model must be on the same device")
    if len(model.blocks) == 0:
        raise ValueError("this cache representation requires at least one block")
    B, C = ids.shape

    if caches is None:
        caches = [None] * len(model.blocks)
        P = 0
    else:
        if not isinstance(caches, (list, tuple)) or len(caches) != len(model.blocks):
            raise ValueError("provide one complete KV pair per layer")
        P = None
        for block, pair in zip(model.blocks, caches):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("each layer cache must be a (K, V) pair")
            Kpast, Vpast = pair
            if not all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in pair):
                raise ValueError("cached tensors must have shape (B, heads, P, d)")
            if P is None:
                P = Kpast.shape[-2]
            expected = (B, block.attn.heads, P, block.attn.head_dim)
            if tuple(Kpast.shape) != expected or tuple(Vpast.shape) != expected:
                raise ValueError("cache batch, heads, prefix lengths, or widths disagree")
            weight = block.attn.qkv.weight
            if any(t.device != weight.device or t.dtype != weight.dtype for t in pair):
                raise ValueError("cache device/dtype must match its projection weights")
    if P + C > model.max_length:
        raise ValueError("position range exceeded")

    positions = torch.arange(P, P + C, device=ids.device)
    x = model.token(ids) + model.position(positions)[None]
    new_caches = []
    for block, previous in zip(model.blocks, caches):
        z = block.norm1(x)
        Q, K, V = block.attn.qkv(z).chunk(3, dim=-1)
        h, d = block.attn.heads, block.attn.head_dim
        def split(tensor):
            return tensor.reshape(B, C, h, d).transpose(1, 2)
        Q, K, V = split(Q), split(K), split(V)
        if previous is not None:
            K = torch.cat([previous[0], K], dim=-2)
            V = torch.cat([previous[1], V], dim=-2)
        query_positions = torch.arange(P, P + C, device=ids.device)
        key_positions = torch.arange(P + C, device=ids.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        read = F.scaled_dot_product_attention(Q, K, V, attn_mask=allowed,
                                             dropout_p=0.0, is_causal=False)
        x = x + block.attn.out(read.transpose(1, 2).reshape(B, C, -1))
        x = x + block.ffn(block.norm2(x))
        new_caches.append((K, V))
    return model.head(model.final_norm(x)), new_caches
