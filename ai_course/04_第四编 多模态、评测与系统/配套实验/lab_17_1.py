import torch
from eval_core import metrics, auroc, average_precision, reliability
from multimodal_core import setup, save_figure

def main():
    setup(171)
    y = torch.tensor([1,0,1,0])
    p = torch.tensor([.9,.8,.4,.1], dtype=torch.float64)
    m = metrics(y,p)
    assert (m["tp"],m["fp"],m["fn"],m["tn"]) == (1,1,1,1)
    assert auroc(y,p) == .75
    assert abs(average_precision(y,p)-5/6) < 1e-12
    assert auroc([1,0],[.5,.5]) == .5
    assert auroc([1,1],[.1,.9]) is None
    assert metrics([0,1],[.1,.2])["precision"] is None
    assert abs(average_precision([1,0],[.5,.5])-.5) < 1e-12
    for bad_y,bad_p in [([],[]),([0],[1.2]),([2],[.5]),([0],[float("nan")]),([0,1],[.5]),([[0]],[[.5]])]:
        try:
            metrics(bad_y,bad_p)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid metric input accepted")

    for invalid_threshold in [float("nan"),float("inf"),-.1,1.1,True]:
        try:
            metrics(y,p,invalid_threshold)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid threshold accepted")
    for invalid_bins in [0,2.5,True]:
        try:
            reliability(y,p,invalid_bins)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid bins accepted")
    score = torch.tensor([9.,8.,4.,1.],dtype=torch.float64)
    assert auroc(y,score)==.75 and abs(average_precision(y,score)-5/6)<1e-12
    assert auroc([0,1],[-100.,200.])==1
    assert average_precision([0,0],[.2,.8]) is None
    assert metrics([0,0],[0.,0.])["f1"] is None
    assert reliability([0,1],[0.,1.],bins=2)[0] == 0
    # Coarse ECE can hide large probability errors within one bin.
    ece_one,_=reliability([0,1],[.9,.1],bins=1)
    assert ece_one==0 and abs(metrics([0,1],[.9,.1])["brier"]-.81)<1e-12
    # Fit temperature on a dedicated validation fixture; evaluate independent test.
    g = torch.Generator().manual_seed(17)
    val_z = torch.randn(600, generator=g)
    val_y = torch.bernoulli(val_z.sigmoid(), generator=g).long()
    test_z = torch.randn(600, generator=g)
    test_y = torch.bernoulli(test_z.sigmoid(), generator=g).long()
    candidates = torch.tensor([.5,1.,2.,3.,4.])
    nll = torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(3*val_z/t, val_y.float())
                       for t in candidates])
    temp = candidates[nll.argmin()]
    raw, calibrated = (3*test_z).sigmoid(), (3*test_z/temp).sigmoid()
    raw_m, cal_m = metrics(test_y,raw), metrics(test_y,calibrated)
    e1, r1 = reliability(test_y,raw)
    e2, r2 = reliability(test_y,calibrated)
    print("hand fixture:",m,"AUROC",auroc(y,p),"AP",average_precision(y,p))
    print(f"validation-selected T={temp}; test Brier {raw_m['brier']:.4f}->{cal_m['brier']:.4f}; ECE {e1:.4f}->{e2:.4f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot([0,1],[0,1],"--",color="gray",label="ideal")
    for rows,label in [(r1,"Raw"),(r2,"Temperature-scaled")]:
        ax.plot([r[1] for r in rows],[r[2] for r in rows],"o-",label=label)
    ax.set(xlabel="Mean predicted P(y=1)",ylabel="Observed positive rate",title="Independent synthetic test set")
    ax.legend()
    fig.tight_layout()
    save_figure(fig,"17_calibration.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
