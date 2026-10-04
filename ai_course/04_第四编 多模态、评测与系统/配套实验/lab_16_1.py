import torch
from torch import nn
from multimodal_core import setup

def main():
    setup(161)
    increasing = torch.tensor([[0.], [1.], [2.], [3.]])
    decreasing = increasing.flip(0)
    torch.testing.assert_close(increasing.mean(0), decreasing.mean(0))
    assert (increasing[-1]-increasing[0]).item() == 3
    assert (decreasing[-1]-decreasing[0]).item() == -3
    layer = nn.MultiheadAttention(4, 2, dropout=0, batch_first=True).eval()
    x = torch.randn(1, 6, 4)
    changed = x.clone()
    changed[:, 4:] += 30
    causal = torch.ones(6, 6, dtype=torch.bool).triu(1)
    a = layer(x, x, x, attn_mask=causal, need_weights=False)[0]
    b = layer(changed, changed, changed, attn_mask=causal, need_weights=False)[0]
    torch.testing.assert_close(a[:, :4], b[:, :4])
    unsafe_a = layer(x, x, x, need_weights=False)[0]
    unsafe_b = layer(changed, changed, changed, need_weights=False)[0]
    assert (unsafe_a[:, :4] - unsafe_b[:, :4]).abs().max() > 1e-4
    # EMA onset detector: no future inputs, fixed threshold chosen for illustration.
    signal = torch.tensor([0.,0.,0.,0.,1.,1.,1.,1.])
    state, first = 0., None
    for t, value in enumerate(signal):
        state = 0.5 * state + 0.5 * value.item()
        if state >= 0.7 and first is None:
            first = t
    assert first == 5

    # Centered preprocessing leaks x[t+1] into feature[t] before any attention.
    raw = torch.tensor([0.,1.,2.,3.,4.,5.])
    future_changed = raw.clone()
    future_changed[4:] += 100
    center = lambda z: (z[:-2]+z[1:-1]+z[2:])/3
    assert center(raw)[2] != center(future_changed)[2]  # feature centered at t=3
    events, alarms = [(4,6),(10,12)], [5,6,9]
    matched, false_alarms, delays = set(), 0, []
    for alarm in alarms:
        eligible = [i for i,(start,end) in enumerate(events)
                    if i not in matched and start <= alarm <= end]
        if eligible:
            index = eligible[0]
            matched.add(index)
            delays.append(alarm-events[index][0])
        else:
            false_alarms += 1
    assert (len(matched),false_alarms,len(events)-len(matched),delays)==(1,2,1,[1])
    print("event scoring: TP=1 FP=2 FN=1; centered-preprocessing leakage reproduced")
    print("same mean/opposite order; causal perturbation test passed")

    print("toy onset=4, first detection=5, delay=1 sample; no trained video model")

if __name__ == "__main__":
    main()
