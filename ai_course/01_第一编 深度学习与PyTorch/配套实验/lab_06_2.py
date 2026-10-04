"""Executable examples for lesson 06.2; run from any working directory."""

import torch
from torch import nn

torch.manual_seed(13)
torch.set_num_threads(1)
B, T, D, H = 2, 5, 3, 4
X = torch.randn(B, T, D, dtype=torch.float64)
rnn = nn.RNN(D, H, nonlinearity="tanh", batch_first=True).double()

h = torch.zeros(B, H, dtype=torch.float64)
states = []
for t in range(T):
    h = torch.tanh(X[:, t] @ rnn.weight_ih_l0.T
                   + h @ rnn.weight_hh_l0.T
                   + rnn.bias_ih_l0 + rnn.bias_hh_l0)
    states.append(h)
manual = torch.stack(states, dim=1)
actual, final = rnn(X)
torch.testing.assert_close(manual, actual)
torch.testing.assert_close(h, final[0])

cell = nn.LSTMCell(D, H).double()
x = X[:, 0]
h0 = torch.randn(B, H, dtype=torch.float64)
c0 = torch.randn(B, H, dtype=torch.float64)
gates = x @ cell.weight_ih.T + h0 @ cell.weight_hh.T + cell.bias_ih + cell.bias_hh
qi, qf, qg, qo = gates.chunk(4, dim=-1)
i, f, candidate, o = qi.sigmoid(), qf.sigmoid(), qg.tanh(), qo.sigmoid()
c1 = f * c0 + i * candidate
h1 = o * c1.tanh()
reference_h, reference_c = cell(x, (h0, c0))
torch.testing.assert_close(h1, reference_h)
torch.testing.assert_close(c1, reference_c)

for a in [0.8, 1.0, 1.2]:
    initial = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    state = initial
    for _ in range(30):
        state = a * state
    state.backward()
    print("a, measured gradient, exact:", a, initial.grad.item(), a**30)

import torch
from torch import nn

torch.manual_seed(21)
torch.set_num_threads(1)
X = torch.randn(240, 8, 1)
y = (X[:, 0, 0] > 0).long()
train_x, val_x, train_y, val_y = X[:180], X[180:], y[:180], y[180:]

class RememberFirst(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])

model = RememberFirst()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()
for _ in range(160):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(train_x), train_y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
model.eval()
with torch.no_grad():
    accuracy = (model(val_x).argmax(-1) == val_y).float().mean().item()
print("remember-first validation accuracy:", accuracy)

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

output = Path(__file__).resolve().parent / "figures"
output.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
steps = list(range(61))
for factor in [0.8, 1.0, 1.2]:
    ax.semilogy(steps, [factor**t for t in steps], label=f"a = {factor}")
ax.set(xlabel="Sequence length T", ylabel="Gradient magnitude",
       title="Scalar recurrence: dh_T / dh_0 = a^T")
ax.legend()
fig.tight_layout()
fig.savefig(output / "06_2_gradient_paths.png", dpi=150)
plt.close(fig)


# Manual batched BPTT, including a direct probe loss at every time step.
torch.manual_seed(602)
B,T,D,H = 2,4,3,5
X = torch.randn(B,T,D,dtype=torch.float64,requires_grad=True)
h0 = torch.randn(B,H,dtype=torch.float64,requires_grad=True)
wx = torch.randn(H,D,dtype=torch.float64,requires_grad=True)
wh = (torch.randn(H,H,dtype=torch.float64)*.2).requires_grad_()
bias = torch.randn(H,dtype=torch.float64,requires_grad=True)
probe = torch.randn(B,T,H,dtype=torch.float64)
states = [h0]
for t in range(T):
    states.append(torch.tanh(X[:,t] @ wx.T+states[-1] @ wh.T+bias))
hidden = torch.stack(states[1:],1)
(hidden*probe).sum().backward()
gx,gwx,gwh,gb = (torch.zeros_like(X),torch.zeros_like(wx),
                  torch.zeros_like(wh),torch.zeros_like(bias))
future = torch.zeros(B,H,dtype=torch.float64)
with torch.no_grad():
    for t in reversed(range(T)):
        delta = (probe[:,t]+future)*(1-states[t+1].square())
        gwx += delta.T @ X[:,t]
        gwh += delta.T @ states[t]
        gb += delta.sum(0)
        gx[:,t] = delta @ wx
        future = delta @ wh
for actual,expected in ((X.grad,gx),(wx.grad,gwx),(wh.grad,gwh),(bias.grad,gb),(h0.grad,future)):
    torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-9)

# Padding updates an RNN state; packing recovers each unpadded final state.
from torch.nn.utils.rnn import pack_padded_sequence
sequence = torch.randn(2,5,3,dtype=torch.float64)
lengths = torch.tensor([5,2])
sequence[1,2:] = 0
recurrent = nn.RNN(3,4,batch_first=True).double().eval()
with torch.no_grad():
    _,naive_final = recurrent(sequence)
    packed = pack_padded_sequence(sequence,lengths,batch_first=True,enforce_sorted=False)
    _,packed_final = recurrent(packed)
    for n,length in enumerate(lengths):
        _,individual = recurrent(sequence[n:n+1,:int(length)])
        torch.testing.assert_close(packed_final[:,n:n+1],individual)
    assert not torch.allclose(naive_final[:,1],packed_final[:,1])
# Detach preserves the forward value but cuts the earlier derivative path.
old = torch.tensor(2.,requires_grad=True)
carried = 3*old
next_leaf = carried.detach().requires_grad_()
(next_leaf.square()).backward()
assert old.grad is None
torch.testing.assert_close(next_leaf.grad,torch.tensor(12.))
print("manual BPTT, packed final states, and detach boundary verified")


# Compare the differentiable packed LSTM gate computation, not just its values.
torch.manual_seed(603)
lstm_cell = nn.LSTMCell(3,4).double()
lx = torch.randn(2,3,dtype=torch.float64,requires_grad=True)
lh = torch.randn(2,4,dtype=torch.float64,requires_grad=True)
lc = torch.randn(2,4,dtype=torch.float64,requires_grad=True)
packed_gates = lx @ lstm_cell.weight_ih.T + lh @ lstm_cell.weight_hh.T
packed_gates = packed_gates + lstm_cell.bias_ih + lstm_cell.bias_hh
li,lf,lg,lo = packed_gates.chunk(4,-1)
manual_c = lf.sigmoid()*lc + li.sigmoid()*lg.tanh()
manual_h = lo.sigmoid()*manual_c.tanh()
ref_h,ref_c = lstm_cell(lx,(lh,lc))
probe_h,probe_c = torch.randn_like(lh),torch.randn_like(lc)
variables = (lx,lh,lc,*lstm_cell.parameters())
manual_grads = torch.autograd.grad((manual_h*probe_h).sum()+(manual_c*probe_c).sum(),variables)
reference_grads = torch.autograd.grad((ref_h*probe_h).sum()+(ref_c*probe_c).sum(),variables)
for left,right in zip(manual_grads,reference_grads):
    torch.testing.assert_close(left,right,atol=1e-11,rtol=1e-9)
print("LSTM gate inputs, states, and all parameter gradients match PyTorch")
