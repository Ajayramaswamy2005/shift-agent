"""Manager approval: pause, survive restart, resume the right conversation."""
from tests.conftest import TODAY, at, intent


def _escalate(h, emp="G201", time="19:30"):
    at(15, 0)  # inside Globex's 4 h cut-off
    r = h.say(emp, f"drop at {time} today", intent("change_time", "drop", TODAY, time))
    assert r["status"] == "rule_violation"
    r = h.say(emp, "send it to my manager", intent("accept_offer", choice="manager"))
    assert r["status"] == "pending_approval"
    return r["request_id"]


def test_approval_survives_restart_and_resumes_right_conversation(h):
    rid = _escalate(h, "G201")
    _escalate(h, "G202")                       # a second paused conversation in the same company
    assert h.slot("G201") == "18:30"
    h.restart()                                # service restarts while both wait
    assert h.decide("M200", rid, approved=True).json()["outcome"] == "approved"
    assert h.slot("G201") == "19:30"           # the right employee moved
    assert h.slot("G202") == "18:30"           # the other paused one is still waiting
    assert len(h.requests("pending_approval")) == 1
    # The employee sees the decision on their next message.
    r = h.say("G201", "any news?", intent("status"))
    assert any("approved" in u for u in r["updates"])
    assert h.requests("applied")[0]["decided_by"] == "M200"


def test_rejection_keeps_trip_and_tells_employee(h):
    rid = _escalate(h)
    h.restart()
    out = h.decide("M200", rid, approved=False, note="No escort available that late").json()
    assert out["outcome"] == "declined" and "No escort available" in out["employee_message"]
    assert h.slot("G201") == "18:30"
    assert h.requests("rejected")[0]["decided_by"] == "M200"


def test_message_while_waiting_gets_plain_status_without_llm(h):
    _escalate(h)
    calls = h.llm.calls
    r = h.say("G201", "hello?? any update")
    assert r["status"] == "pending_approval" and "still waiting" in r["reply"]
    assert h.llm.calls == calls and r["meta"]["llm_calls"] == 0


def test_cab_fills_before_approval(h):
    rid = _escalate(h)
    h.q("UPDATE cabs SET booked = capacity WHERE id = 'GLX-20261119-D-1930' RETURNING id")
    out = h.decide("M200", rid, approved=True).json()
    assert out["outcome"] == "failed" and "filled up" in out["employee_message"]
    assert h.slot("G201") == "18:30"
    assert h.requests("failed")


def test_decision_cannot_be_applied_twice(h):
    rid = _escalate(h)
    assert h.decide("M200", rid, approved=True).status_code == 200
    assert h.decide("M200", rid, approved=False).status_code == 409
    assert h.slot("G201") == "19:30"


def test_double_escalation_creates_one_request(h):
    _escalate(h)
    # The conversation is paused; a repeated "send to manager" just reports the status.
    r = h.say("G201", "send to manager", intent("accept_offer", choice="manager"))
    assert r["status"] == "pending_approval"
    assert len(h.requests()) == 1
