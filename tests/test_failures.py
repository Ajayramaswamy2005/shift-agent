"""When things go wrong the employee gets a plain message, never a stack trace or a hang."""
from app.llm import LLMError
from tests.conftest import TODAY, intent


def _plain(r):
    return "Traceback" not in r["reply"] and "Error" not in r["reply"] and "psycopg" not in r["reply"]


def test_roster_tool_timeout(h):
    h.tools.faults["get_upcoming_trips"] = "timeout"
    r = h.say("E101", "drop at 19:00")
    assert r["status"] == "error" and "nothing was changed" in r["reply"] and _plain(r)
    assert h.llm.calls == 0          # no point calling the LLM without the roster


def test_cab_options_tool_error(h):
    h.tools.faults["find_cab_options"] = "error"
    r = h.say("E101", "drop at 19:00", intent("change_time", "drop", TODAY, "19:00"))
    assert r["status"] == "error" and "+91-80-4000-1000" in r["reply"] and _plain(r)
    assert h.slot("E101") == "18:30"


def test_apply_tool_timeout_changes_nothing(h):
    h.tools.faults["apply_change"] = "timeout"
    r = h.say("E101", "drop at 19:00", intent("change_time", "drop", TODAY, "19:00"))
    assert r["status"] == "error" and h.slot("E101") == "18:30" and h.requests() == []


def test_llm_down(h):
    r = h.say("E101", "drop at 19:00", LLMError("quota exceeded"))
    assert r["status"] == "error" and "trouble understanding" in r["reply"] and _plain(r)


def test_llm_returns_garbage_twice(h):
    r = h.say("E101", "drop at 19:00", '{"intent": "book_helicopter"}', "```not json```")
    assert r["status"] == "error" and r["meta"]["llm_calls"] == 2 and h.slot("E101") == "18:30"


def test_llm_invents_bad_time(h):
    r = h.say("E101", "drop at 25:99", intent("change_time", "drop", TODAY, "25:99"),
              intent("change_time", "drop", TODAY, None))
    assert r["status"] == "clarify"   # invalid time rejected, repaired output asks for the time


def test_cab_fills_while_employee_decides(h):
    r = h.say("E101", "8 pm drop", intent("change_time", "drop", TODAY, "20:00"))
    offered = "19:30" if "19:30" in r["reply"] else "20:30"
    h.q("UPDATE cabs SET booked = capacity WHERE id = %s RETURNING id", f"ACM-20261119-D-{offered.replace(':', '')}")
    r = h.say("E101", "yes", intent("accept_offer"))
    assert r["status"] == "unavailable" and "full" in r["reply"]   # re-checked live, not trusted from before
    assert h.slot("E101") == "18:30"


def test_unexpected_crash_returns_plain_500(h, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(h.client.app.state.agent, "handle_message", boom)
    r = h.client.post("/chat", json={"message": "hi"}, headers=h.auth("E101"))
    assert r.status_code == 500
    assert "secret internal detail" not in r.text and "Traceback" not in r.text
    assert "nothing was changed" in r.json()["reply"]


def test_bad_input(h):
    for body in ({}, {"message": ""}, {"message": "x" * 1001}, {"msg": "hi"}):
        r = h.client.post("/chat", json=body, headers=h.auth("E101"))
        assert r.status_code == 422 and "reply" in r.json()
    r = h.decide("M100", "not-a-uuid", True)
    assert r.status_code == 404
