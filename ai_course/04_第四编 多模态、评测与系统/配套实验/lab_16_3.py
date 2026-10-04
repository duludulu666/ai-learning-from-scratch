import torch
from multimodal_core import setup
from eval_core import auroc, selective_risk

def main():
    setup(163)
    # Known classes lie near (-1,0), (1,0). A classifier ignores the second coordinate.
    known = torch.tensor([[-1.,0.],[1.,0.]])
    shifted = torch.tensor([[-1.,20.],[1.,20.]])
    confidence = lambda x: torch.stack([-5*x[:,0], 5*x[:,0]], dim=-1).softmax(-1).max(-1).values
    torch.testing.assert_close(confidence(known), confidence(shifted))
    conf = torch.cat([confidence(known), confidence(shifted)])
    labels_ood = torch.tensor([0,0,1,1])
    score = 1-conf
    assert auroc(labels_ood, score) == .5
    prototypes = known
    distance = torch.cdist(torch.cat([known,shifted]), prototypes).min(1).values
    assert auroc(labels_ood, distance) == 1.  # Ranking accepts raw distances, not only probabilities.
    correct = torch.tensor([1,1,0,1,0], dtype=torch.float32)
    c = torch.tensor([.99,.90,.85,.70,.55])
    for threshold in [.5,.8,.95,1.]:
        coverage,risk = selective_risk(correct,c,threshold)
        print(f"threshold={threshold:.2f} coverage={coverage:.2f} risk={risk}")
    assert selective_risk(correct,c,1.) == (0.,None)
    # Worse confidence ordering: stricter thresholds can increase selective risk.
    assert selective_risk([0,1],[.99,.8],.95)[1] > selective_risk([0,1],[.99,.8],.5)[1]
    p_image, p_text = torch.tensor([.95,.05]), torch.tensor([.05,.95])
    mean = (p_image+p_text)/2
    assert mean.max() == .5
    print("max-softmax OOD AUROC=0.5; distance AUROC=1 on this constructed fixture only")
    print("contradictory two-class evidence averages to:", mean.tolist())

if __name__ == "__main__":
    main()
