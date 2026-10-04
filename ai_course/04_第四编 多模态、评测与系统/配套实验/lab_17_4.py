from dataclasses import dataclass
import math
from numbers import Real

@dataclass(frozen=True)
class Case:
    case_id: str
    slice: str
    truth: str

def score(cases, outputs):
    if not cases:
        raise ValueError("Empty evaluation set")
    if any(not isinstance(c,Case) or any(not isinstance(v,str) or not v for v in (c.case_id,c.slice,c.truth)) for c in cases):
        raise ValueError("Cases need nonempty string IDs, slices and references")
    if not isinstance(outputs,dict):
        raise ValueError("Outputs must map case IDs to fixtures")
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Duplicate case ID")
    if set(outputs) != {c.case_id for c in cases}:
        raise ValueError("Missing or unexpected output IDs")
    rows = []
    for c in cases:
        record = outputs[c.case_id]
        if not isinstance(record,(tuple,list)) or len(record)!=2:
            raise ValueError("Each output must be (answer, latency)")
        answer, latency = record
        if (not isinstance(answer,str) or isinstance(latency,bool) or not isinstance(latency,Real)
                or not math.isfinite(latency) or latency < 0):
            raise ValueError("Invalid output fixture")
        rows.append((c.slice, answer == c.truth, latency))
    slices = {s: sum(ok for q,ok,_ in rows if q==s)/sum(q==s for q,_,_ in rows)
              for s,_,_ in rows}
    return {"accuracy":sum(ok for _,ok,_ in rows)/len(rows),
            "slices":slices,"mean_ms":sum(t for _,_,t in rows)/len(rows)}

def main():
    cases = [Case("a","common","yes"),Case("b","common","no"),Case("c","common","yes"),
             Case("d","critical","abstain")]
    base = {"a":("yes",10.),"b":("yes",10.),"c":("no",10.),"d":("abstain",10.)}
    candidate = {"a":("yes",12.),"b":("no",12.),"c":("yes",12.),"d":("yes",12.)}
    a,b = score(cases,base),score(cases,candidate)
    promote = (b["accuracy"]>=a["accuracy"] and
               b["slices"]["critical"]>=a["slices"]["critical"] and b["mean_ms"]<=15)
    assert b["accuracy"]>a["accuracy"] and not promote
    # Evaluator unit tests are separate from evaluating a model.
    try:
        score(cases,{"a":("yes",1.)})
    except ValueError:
        pass
    else:
        raise AssertionError("Missing cases accepted")
    for bad_cases,bad_outputs in [
        ([], {}),
        ([cases[0], cases[0]], {"a":("yes",1.)}),
        ([cases[0]], {"a":("yes",float("nan"))}),
        ([cases[0]], {"a":("yes",-1.)}),
        ([cases[0]], {"a":("yes",True)}),
        ([cases[0]], {"a":("yes","fast")}),
        ([cases[0]], {"a":("yes",)}),
        ([cases[0]], {"a":("yes",1.),"extra":("yes",1.)}),
        ([Case("","common","yes")], {"":("yes",1.)}),
    ]:
        try:
            score(bad_cases,bad_outputs)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid evaluation accepted")
    print("baseline:",a,"candidate:",b,"promote:",promote)
    print("Outputs and latency values are hand-authored fixtures, not API calls or measured model latency.")

if __name__ == "__main__":
    main()
