"""Retrieval evaluation with a declared lexical baseline and toy query expansion."""
import json
from app_core import (Retriever,authorized_docs,extractive_answer,validate_answer,DOCUMENTS)

docs=authorized_docs("guest")
baseline=Retriever(docs,expand=False)
expanded=Retriever(docs,expand=True)
cases=[
    ("book loans","public-loans"),
    ("opening time","public-hours"),
    ("renewals extension","public-renew"),
    ("printing costs","public-print"),
    ("borrow","public-loans"),
    ("opens","public-hours"),
]
def metrics(index):
    hits1,hits3,rr=0,0,0.
    for query,target in cases:
        ranked=index.search(query,3);ids=[doc["id"] for doc,_ in ranked]
        hits1+=int(bool(ids) and ids[0]==target)
        hits3+=int(target in ids)
        rr+=1/(ids.index(target)+1) if target in ids else 0
        print("query/result:",query,ids)
    return hits1/len(cases),hits3/len(cases),rr/len(cases)
base_metrics=metrics(baseline)
expanded_metrics=metrics(expanded)
assert expanded_metrics[0]>base_metrics[0]
assert expanded_metrics[0]==1.
assert not expanded.search("quantum superconductivity")
answer=extractive_answer("book loans",expanded.search("book loans"))
validate_answer(json.dumps(answer),[d["id"] for d in docs])
assert answer["answer"]=="14 days"
unknown=extractive_answer("quantum superconductivity",[])
assert unknown["abstain"]
# Access control is enforced before indexing; private rows never enter guest candidates.
assert all(d["id"]!="staff-access" for d in docs)
assert all(d["id"]!="staff-access" for d,_ in expanded.search("staff book loans",10))
# A valid citation is not a proof of entailment; this exact fixture support check is narrow.
source={d["id"]:d for d in docs}
def fixture_supported(answer):
    return (not answer["abstain"] and len(answer["citations"])==1 and
            answer["answer"]==source[answer["citations"][0]]["answer"])
assert fixture_supported(answer)
assert not fixture_supported(dict(answer="999 days",citations=["public-loans"],abstain=False))
print("baseline recall@1/recall@3/MRR:",base_metrics)
print("expanded recall@1/recall@3/MRR:",expanded_metrics)
print("no-hit abstention, citation existence, fixture support and pre-index ACL passed")
print("Query expansion is hand-designed on this tiny fixture; no independent learned-retrieval benchmark.")


from app_core import guarded_fixture_answer
# Lexical overlap can confidently retrieve a fact that does not answer the question.
tricky="book loans quantum superconductivity"
ranked=expanded.search(tricky)
naive=extractive_answer(tricky,ranked)
assert not naive["abstain"]  # A documented failure, not a desired success.
assert guarded_fixture_answer(tricky,ranked)["abstain"]
assert guarded_fixture_answer("book loans",expanded.search("book loans"))["answer"]=="14 days"
print("lexically matched unanswerable question / naive answer:",tricky,naive)
print("strict fixture-intent gate abstains; it deliberately rejects unknown paraphrases")
# Retrieval snapshots/results cannot be mutated to rewrite the index or source registry.
retrieved=expanded.search("book loans")
retrieved[0][0]["answer"]="999 days"
assert expanded.search("book loans")[0][0]["answer"]=="14 days"
docs[0]["answer"]="999 days"
assert authorized_docs("guest")[0]["answer"]=="14 days"
assert expanded.search("book loans")[0][0]["answer"]=="14 days"
# Hand-computed TF-IDF and cosine, independent from the implementation's dot loop.
import math
short=Retriever([dict(id="a",text="red fox"),dict(id="b",text="blue owl")])
score=short.search("red")[0][1]
assert math.isclose(score,1/math.sqrt(2))
assert Retriever([]).search("anything")==[]
# Multi-relevant metric example distinguishes recall from reciprocal rank.
relevant={"a","b"};returned=["c","b"]
assert len(relevant&set(returned))/len(relevant)==.5
assert 1/(returned.index("b")+1)==.5
print("known-overlap abstention, snapshot isolation, hand cosine and metric examples passed")
