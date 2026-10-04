import math
from numbers import Real

def gate(proposal, allowed_read_ids, allowed_write_ids, approvals):
    """Read-only policy fixture: no action is executed.
    Trusted approval tuples bind (action, resource, exact payload), not confidence.
    This models matching, not authentication, persistence, or replay protection.
    """
    if not isinstance(proposal,dict):
        return "invalid"
    action=proposal.get("action")
    if not isinstance(action,str):
        return "invalid"
    required={"action","resource","confidence"} | ({"payload"} if action=="write" else set())
    if set(proposal)!=required:
        return "invalid"
    if not isinstance(proposal["resource"],str) or not proposal["resource"]:
        return "invalid"
    c=proposal["confidence"]
    if isinstance(c,bool) or not isinstance(c,Real):
        return "invalid"
    if not 0<=c<=1 or not math.isfinite(c):
        return "invalid"
    resource=proposal["resource"]
    if action=="read":
        return "allow" if resource in allowed_read_ids else "deny"
    if action=="write":
        payload=proposal["payload"]
        if not isinstance(payload,str):
            return "invalid"
        if resource not in allowed_write_ids:
            return "deny"
        request=(action,resource,payload)
        return "allow" if request in approvals else "approval_required"
    return "deny"

def main():
    reads={"public_note"}
    writes={"public_note"}
    approvals={("write","public_note","reviewed text")}
    fixtures=[
        ({"action":"read","resource":"public_note","confidence":.4},"allow"),
        ({"action":"read","resource":"private_note","confidence":1.},"deny"),
        ({"action":"write","resource":"public_note","confidence":1.,"payload":"new text"},"approval_required"),
        ({"action":"execute","resource":"anything","confidence":1.},"deny"),
        ({"action":"read","resource":"public_note","confidence":float("nan")},"invalid"),
        ({"action":"read","resource":"public_note","confidence":.9,"role":"system"},"invalid"),
        ({"action":"write","resource":"public_note","confidence":.1,"payload":"reviewed text"},"allow"),
        ({"action":"write","resource":"public_note","confidence":1.,"payload":"changed text"},"approval_required"),
        ({"action":"write","resource":"private_note","confidence":1.,"payload":"reviewed text"},"deny"),
        ({"action":"read","resource":"public_note","confidence":True},"invalid"),
        ({"action":"write","resource":"public_note","confidence":.9},"invalid"),
        ({"action":"read","resource":"","confidence":.9},"invalid"),
    ]
    for proposal,expected in fixtures:
        assert gate(proposal,reads,writes,approvals)==expected
    # Even an approval tuple cannot expand the separate permitted resource scope.
    off_scope={"action":"write","resource":"private_note","confidence":1.,"payload":"text"}
    assert gate(off_scope,reads,writes,{("write","private_note","text")})=="deny"
    # Confidence is irrelevant to an otherwise authorized action.
    for confidence in [0.,.5,1.]:
        assert gate({"action":"read","resource":"public_note","confidence":confidence},
                    reads,writes,approvals)=="allow"
    n=100
    upper=1-.05**(1/n)
    assert .029<upper<.030
    concise={"text":"The observed shape is red.","supported_claims":1,"claims":1}
    padded={"text":"The observed shape is red. It is famous, expensive, rare, and imported.",
            "supported_claims":1,"claims":5}
    assert len(padded["text"])>len(concise["text"])
    assert padded["supported_claims"]/padded["claims"]<concise["supported_claims"]/concise["claims"]
    print("12 proposal fixtures plus scope/approval/confidence invariants passed; no tool execution")
    print(f"zero failures / {n} IID trials: one-sided 95% failure-rate upper bound={upper:.5f}")
    print("Length proxy improved while support precision declined; fixture, not a trained reward model.")

if __name__=="__main__":
    main()
