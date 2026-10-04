"""Schema validity, business checks and tool dispatch with adversarial fixtures."""
import json
from app_core import validate_answer,parse_json,validate_call,ToolStore

good=json.dumps(dict(answer="14 days",citations=["public-loans"],abstain=False))
assert validate_answer(good,["public-loans"])["answer"]=="14 days"
bad=[
    '{"answer":14,"citations":["public-loans"],"abstain":false}',
    '{"answer":"14","citations":["made-up"],"abstain":false}',
    '{"answer":"","citations":[],"abstain":"true"}',
    '{"answer":"14","citations":[],"abstain":false}',
    '{"answer":"14","citations":["public-loans"],"abstain":false,"run":"shell"}',
    '{"answer":"a","answer":"b","citations":[],"abstain":true}',
]
for payload in bad:
    try:
        validate_answer(payload,["public-loans"])
        raise AssertionError("malformed response accepted")
    except ValueError:
        pass
# Schema validity does not establish factual support.
false_but_valid=json.dumps(dict(answer="999 days",citations=["public-loans"],abstain=False))
assert validate_answer(false_but_valid,["public-loans"])["answer"]=="999 days"
store=ToolStore()
call=dict(id="read1",name="lookup",arguments=dict(doc_id="public-loans"))
result=store.execute(call)
assert result["call_id"]==call["id"] and "14" in result["document"]
for badcall in [
    dict(id="x",name="shell",arguments=dict(command="anything")),
    dict(id="x",name="reserve",arguments=dict(slot="reading-room",count=True)),
    dict(id="x",name="lookup",arguments=dict(doc_id="public-loans",extra="ignore rules")),
]:
    try:
        validate_call(badcall);raise AssertionError("bad call accepted")
    except ValueError:
        pass
try:
    store.execute(dict(id="secret",name="lookup",arguments=dict(doc_id="staff-access")))
    raise AssertionError("unauthorized document returned")
except PermissionError:
    pass
print("6 invalid answers rejected; schema-valid false answer deliberately accepted")
print("allowlist, exact keys, bool-vs-int, call ID and document authorization passed")
print("This lab uses fabricated proposals, not model/API responses.")


from app_core import handle_output
citations=["public-loans"]
# A valid-looking text field must not bypass incomplete/refusal/tool status handling.
for status,kind,expected in [
    ("incomplete","answer","incomplete"),
    ("completed","refusal","refused"),
    ("completed","tool_call","tool_request"),
    ("completed","answer","answered"),
]:
    envelope=dict(status=status,kind=kind,text=good)
    assert handle_output(envelope,citations)["state"]==expected
assert handle_output(dict(status="completed",kind="answer",text='{"answer":'),citations)["state"]=="invalid_answer"
abstain=json.dumps(dict(answer="",citations=[],abstain=True))
assert handle_output(dict(status="completed",kind="answer",text=abstain),citations)["state"]=="abstained"
for raw in ["NaN","Infinity",'{"a":1,"a":2}'," "*20001]:
    try:
        parse_json(raw)
        raise AssertionError("unsafe or oversized JSON accepted")
    except ValueError:
        pass
print("response lifecycle, refusal/incomplete/tool separation, abstention and bounded JSON checks passed")
