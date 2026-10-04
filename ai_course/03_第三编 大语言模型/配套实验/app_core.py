"""Offline application fixtures: validation and retrieval, not an LLM or API SDK."""
import json
import copy
import math
import re
from collections import Counter
from dataclasses import dataclass

DOCUMENTS = [
    dict(id="public-loans", tenant="public", text="Library book loans last 14 days.", answer="14 days"),
    dict(id="public-hours", tenant="public", text="Library opening time is 09:00.", answer="09:00"),
    dict(id="public-renew", tenant="public", text="Library renewals allow one extension of 7 days.", answer="7 days"),
    dict(id="public-print", tenant="public", text="Library printing costs 0.10 credits per page.", answer="0.10 credits"),
    dict(id="staff-access", tenant="staff", text="Staff book loans last 60 days.", answer="60 days"),
]
# All facts are invented teaching fixtures, not claims about an actual institution.
SYNONYMS = {"borrow":"loans","borrowing":"loans","loan":"loans",
            "opens":"opening","open":"opening","renew":"renewals"}

def words(text, expand=False):
    tokens = re.findall(r"[a-z0-9]+",text.lower())
    return [SYNONYMS.get(token,token) for token in tokens] if expand else tokens

def reject_constants(value):
    raise ValueError("nonstandard JSON constant: "+value)

def reject_duplicate_keys(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError("duplicate JSON key")
        result[key]=value
    return result

def parse_json(text):
    if not isinstance(text,str) or len(text)>20000:
        raise ValueError("JSON payload must be a bounded string")
    return json.loads(text, parse_constant=reject_constants, object_pairs_hook=reject_duplicate_keys)

def validate_answer(text,allowed_citations):
    obj=parse_json(text)
    if not isinstance(obj,dict) or set(obj)!={"answer","citations","abstain"}:
        raise ValueError("answer schema keys")
    if not isinstance(obj["answer"],str) or len(obj["answer"])>1000:
        raise ValueError("answer must be a bounded string")
    if type(obj["abstain"]) is not bool:
        raise ValueError("abstain must be bool")
    cites=obj["citations"]
    if not isinstance(cites,list) or any(not isinstance(c,str) for c in cites):
        raise ValueError("citations must be a string list")
    if len(cites)!=len(set(cites)) or not set(cites)<=set(allowed_citations):
        raise ValueError("unknown or duplicate citations")
    if obj["abstain"] and (obj["answer"] or cites):
        raise ValueError("abstention must have empty answer and citations")
    if not obj["abstain"] and (not obj["answer"] or not cites):
        raise ValueError("answer requires evidence")
    return obj

def handle_output(event,allowed_citations):
    """Provider-neutral response envelope, NOT an OpenAI API event schema."""
    if not isinstance(event,dict) or set(event)!={"status","kind","text"}:
        raise ValueError("response envelope schema")
    if event["status"]=="incomplete":
        return dict(state="incomplete",answer=None)
    if event["status"]!="completed":
        raise ValueError("unknown response status")
    if event["kind"]=="refusal":
        return dict(state="refused",answer=None)
    if event["kind"]=="tool_call":
        return dict(state="tool_request",answer=None)
    if event["kind"]!="answer":
        raise ValueError("unknown response kind")
    try:
        answer=validate_answer(event["text"],allowed_citations)
    except (ValueError,TypeError,RecursionError):
        return dict(state="invalid_answer",answer=None)
    return dict(state="abstained" if answer["abstain"] else "answered",answer=answer)

class Retriever:
    """TF-IDF cosine index fitted on the supplied, already-authorized documents."""
    def __init__(self,documents,expand=False):
        self.documents=copy.deepcopy(list(documents))
        self.expand=expand
        self.counts=[Counter(words(d["text"],expand)) for d in self.documents]
        terms=set().union(*(set(c) for c in self.counts)) if self.counts else set()
        self.idf={t:math.log((1+len(self.documents))/(1+sum(t in c for c in self.counts)))+1
                  for t in terms}
        self.vectors=[{t:count*self.idf[t] for t,count in c.items()} for c in self.counts]

    def search(self,query,k=3):
        if type(k) is not int or k<1: raise ValueError("k must be positive")
        q={t:count*self.idf[t] for t,count in Counter(words(query,self.expand)).items() if t in self.idf}
        qnorm=math.sqrt(sum(v*v for v in q.values()))
        if not qnorm: return []
        scored=[]
        for doc,vector in zip(self.documents,self.vectors):
            norm=math.sqrt(sum(v*v for v in vector.values()))
            score=sum(q.get(t,0)*v for t,v in vector.items())/(qnorm*norm) if norm else 0.
            if score>0: scored.append((doc,score))
        return copy.deepcopy(sorted(scored,key=lambda item:(-item[1],item[0]["id"]))[:k])

def authorized_docs(actor):
    # actor comes from trusted application identity, never model-generated arguments.
    return copy.deepcopy([d for d in DOCUMENTS if d["tenant"] in ("public",actor)])

def extractive_answer(query,ranked,threshold=.2):
    """Hand-coded task fixture, NOT a generative model or general entailment judge."""
    if not ranked or ranked[0][1]<threshold:
        return dict(answer="",citations=[],abstain=True)
    document=ranked[0][0]
    return dict(answer=document["answer"],citations=[document["id"]],abstain=False)

# Explicit intent registry for the FABRICATED task, not semantic understanding.
FIXTURE_INTENTS={
    "book loans":"public-loans", "borrow":"public-loans",
    "opening time":"public-hours", "opens":"public-hours",
    "renewals extension":"public-renew", "printing costs":"public-print",
}
def guarded_fixture_answer(query,ranked,threshold=.2):
    expected=FIXTURE_INTENTS.get(" ".join(query.lower().split()))
    if expected is None:
        return dict(answer="",citations=[],abstain=True)
    for document,score in ranked:
        if document["id"]==expected and score>=threshold:
            return dict(answer=document["answer"],citations=[document["id"]],abstain=False)
    return dict(answer="",citations=[],abstain=True)

def validate_call(call):
    if not isinstance(call,dict) or set(call)!={"id","name","arguments"}:
        raise ValueError("tool call schema")
    if not isinstance(call["id"],str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}",call["id"]):
        raise ValueError("invalid call ID")
    name,args=call["name"],call["arguments"]
    if not isinstance(name,str) or name not in ("lookup","reserve"):
        raise ValueError("tool not in allowlist")
    if not isinstance(args,dict): raise ValueError("arguments must be an object")
    if name=="lookup":
        if set(args)!={"doc_id"} or not isinstance(args["doc_id"],str):
            raise ValueError("lookup arguments")
    else:
        if set(args)!={"slot","count"} or args["slot"]!="reading-room":
            raise ValueError("reservation arguments")
        if type(args["count"]) is not int or not 1<=args["count"]<=2:
            raise ValueError("count must be integer 1 or 2")
    return call

class ToolStore:
    """Only an in-memory simulation. No external booking or file/network mutation."""
    def __init__(self):
        self.reservations=0
        self.cache={}
        self.fail_after_commit_once=False

    def execute(self,call,actor="guest",approvals=None):
        validate_call(call)
        approvals=set() if approvals is None else approvals
        name,args,call_id=call["name"],call["arguments"],call["id"]
        if name=="lookup":
            allowed={d["id"]:d for d in authorized_docs(actor)}
            if args["doc_id"] not in allowed: raise PermissionError("document access denied")
            return dict(call_id=call_id,document=allowed[args["doc_id"]]["text"])
        signature=json.dumps(args,sort_keys=True)
        approval=(actor,call_id,signature)
        if approval not in approvals: raise PermissionError("exact operation needs trusted approval")
        key=(actor,call_id)
        if key in self.cache:
            previous_signature,result=self.cache[key]
            if previous_signature!=signature: raise ValueError("idempotency key reused with different arguments")
            return copy.deepcopy(result)
        self.reservations+=args["count"]
        result=dict(call_id=call_id,reserved=args["count"],slot=args["slot"])
        self.cache[key]=(signature,copy.deepcopy(result))
        if self.fail_after_commit_once:
            self.fail_after_commit_once=False
            raise TimeoutError("simulated response lost after commit")
        return copy.deepcopy(result)

@dataclass
class RunResult:
    status: str
    trace: list
    outputs: list

def run_calls(calls,store,actor="guest",approvals=None,max_calls=3,max_attempts=2,goal_check=None):
    """A scripted proposal queue exercises the controller; it is not a learned planner."""
    if type(max_calls) is not int or max_calls < 1 or type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("call and attempt budgets must be positive integers")
    trace,outputs=[],[]
    for number,call in enumerate(calls):
        if number>=max_calls:
            trace.append(dict(event="budget_exceeded"))
            return RunResult("budget_exceeded",trace,outputs)
        try:
            validate_call(call)
        except (ValueError,TypeError) as exc:
            trace.append(dict(event="invalid_proposal",reason=str(exc)))
            return RunResult("rejected",trace,outputs)
        for attempt in range(1,max_attempts+1):
            try:
                result=store.execute(call,actor,approvals)
                if not isinstance(result,dict) or result.get("call_id")!=call["id"]:
                    trace.append(dict(event="invalid_tool_result",call_id=call["id"]))
                    return RunResult("needs_attention",trace,outputs)
                outputs.append(copy.deepcopy(result))
                trace.append(dict(event="tool_ok",call_id=call["id"],name=call["name"],attempt=attempt))
                break
            except TimeoutError:
                trace.append(dict(event="timeout",call_id=call["id"],attempt=attempt))
                if attempt==max_attempts:
                    return RunResult("needs_attention",trace,outputs)
            except PermissionError as exc:
                trace.append(dict(event="denied",call_id=call["id"],reason=str(exc)))
                return RunResult("needs_approval_or_access",trace,outputs)
            except ValueError as exc:
                trace.append(dict(event="invalid_execution",call_id=call["id"],reason=str(exc)))
                return RunResult("rejected",trace,outputs)
    if goal_check is None:
        return RunResult("operations_completed",trace,outputs)
    verified=goal_check(copy.deepcopy(outputs))
    if type(verified) is not bool:
        raise ValueError("goal checker must return a boolean")
    trace.append(dict(event="goal_check",passed=verified))
    return RunResult("completed" if verified else "goal_not_verified",trace,outputs)
