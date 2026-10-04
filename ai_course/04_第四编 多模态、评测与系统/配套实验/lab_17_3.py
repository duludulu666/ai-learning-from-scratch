import re
import unicodedata
import torch

def canonical(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC",text).casefold()))

def main():
    train = ["The small cat sits.", "A blue square rotates"]
    test = [" THE small CAT sits! ", "A red square rotates"]
    raw_overlap = set(train) & set(test)
    normalized_overlap = set(map(canonical,train)) & set(map(canonical,test))
    assert not raw_overlap and len(normalized_overlap) == 1
    # A group ID must never appear on both sides, even if frame contents differ.
    train_groups, test_groups = {"scene_a","scene_b"}, {"scene_c","scene_d"}
    assert train_groups.isdisjoint(test_groups)
    # Controlled spurious signal: x1 is causal label evidence, x2 changes correlation.
    y = torch.tensor([-1.,1.,-1.,1.])
    x_id = torch.stack([y,y],dim=1)
    x_ood = torch.stack([y,-y],dim=1)
    spurious = lambda x: x[:,1]
    robust = lambda x: x[:,0]
    for name,rule in [("spurious",spurious),("invariant fixture",robust)]:
        print(name,"ID",float((rule(x_id)==y).float().mean()),"shift",float((rule(x_ood)==y).float().mean()))
    # One-dimensional FGSM for an explicitly bounded toy differentiable threat model.
    x = torch.tensor([.1],requires_grad=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(10*x,torch.ones(1))
    loss.backward()
    epsilon = .2
    adv = x.detach() + epsilon*x.grad.sign()
    assert (adv-x.detach()).abs().max() <= epsilon+1e-7
    assert x.item()>0 and adv.item()<0

    assert canonical("C++") == canonical("C"), "A deliberate destructive-normalization collision"
    source=torch.tensor([.8,.2],dtype=torch.float64)
    target=torch.tensor([.2,.8],dtype=torch.float64)
    loss_by_input=torch.tensor([0.,1.],dtype=torch.float64)
    weights=target/source
    torch.testing.assert_close((source*weights*loss_by_input).sum(),(target*loss_by_input).sum())
    print("importance weighting: source risk=.2, target risk=.8, weights=.25/4")
    print("normalized overlaps:",sorted(normalized_overlap),"toy adversarial x:",adv.item())
    print("No poisoning, external attack, or frontier-model robustness test is performed.")

if __name__ == "__main__":
    main()
