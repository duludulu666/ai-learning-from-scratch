"""Runnable examples for lesson 09.3."""

import torch
from torch import nn

torch.manual_seed(29)
torch.set_num_threads(1)
B, S, T, D = 2, 4, 5, 12
transformer = nn.Transformer(
    d_model=D, nhead=3, num_encoder_layers=1, num_decoder_layers=1,
    dim_feedforward=24, dropout=0.0, batch_first=True,
).double().eval()

source = torch.randn(B, S, D, dtype=torch.float64)
target_input = torch.randn(B, T, D, dtype=torch.float64)
source_pad = torch.tensor([[False, False, False, True],
                           [False, False, False, False]])
target_block = torch.ones(T, T, dtype=torch.bool).triu(1)
with torch.no_grad():
    output = transformer(source, target_input, tgt_mask=target_block,
                         src_key_padding_mask=source_pad,
                         memory_key_padding_mask=source_pad)

    changed_target = target_input.clone()
    changed_target[:, 3:] += 10 * torch.randn_like(changed_target[:, 3:])
    changed_output = transformer(source, changed_target, tgt_mask=target_block,
                                 src_key_padding_mask=source_pad,
                                 memory_key_padding_mask=source_pad)
    torch.testing.assert_close(output[:, :3], changed_output[:, :3],
                               atol=1e-9, rtol=1e-7)

    changed_source = source.clone()
    changed_source[0, -1] += 10 * torch.randn_like(changed_source[0, -1])
    padding_output = transformer(changed_source, target_input, tgt_mask=target_block,
                                  src_key_padding_mask=source_pad,
                                  memory_key_padding_mask=source_pad)
    torch.testing.assert_close(output[0], padding_output[0], atol=1e-9, rtol=1e-7)

# Negative controls: show these perturbations are detectable without the masks.
with torch.no_grad():
    unmasked_target = transformer(source, target_input,
                                  src_key_padding_mask=source_pad,
                                  memory_key_padding_mask=source_pad)
    unmasked_changed = transformer(source, changed_target,
                                   src_key_padding_mask=source_pad,
                                   memory_key_padding_mask=source_pad)
    assert not torch.allclose(unmasked_target[:, :3], unmasked_changed[:, :3])
    no_source_masks = transformer(source, target_input, tgt_mask=target_block)
    changed_no_source_masks = transformer(changed_source, target_input, tgt_mask=target_block)
    assert not torch.allclose(no_source_masks[0], changed_no_source_masks[0])
print("masked invariance and unmasked negative controls verified")

assert output.shape == (B, T, D)
labels = torch.tensor([[3, 4, 2, 0], [5, 2, 0, 0]])  # 0=PAD, 1=BOS, 2=EOS
decoder_ids = torch.cat([torch.ones(B, 1, dtype=torch.long), labels[:, :-1]], dim=1)
print("decoder input:", decoder_ids)
print("target labels:", labels)
print("encoder-decoder output:", output.shape)
