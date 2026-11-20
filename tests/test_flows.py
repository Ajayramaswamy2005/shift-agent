"""Happy paths, cut-off rejection, clarification with context, tool selection, LLM call budget."""
from tests.conftest import TODAY, TOMORROW, at, intent


def test_simple_change_applied(h):
    r = h.say("E101", "Can I take the 7 PM drop today?", intent("change_time", "drop", TODAY, "19:00"))
    assert r["status"] == "applied" and "19:00" in r["reply"]
    assert h.slot("E101") == "19:00"
    assert len(h.requests("applied")) == 1
    assert h.q("SELECT booked FROM cabs WHERE id = 'ACM-20261119-D-1830'")[0]["booked"] == 2  # 3 employees were on it, one moved
    assert h.q("SELECT count(*) AS n FROM audit_log WHERE event = 'change_applied'")[0]["n"] == 1


def test_cutoff_rejected_for_globex_but_allowed_for_acme(h):
    at(15, 0)
    g = h.say("G201", "drop at 7:30 pm today", intent("change_time", "drop", TODAY, "19:30"))
    assert g["status"] == "rule_violation"
    assert "Globex India needs changes at least 4 h before" in g["reply"]
    assert "send to manager" in g["reply"]
    assert h.slot("G201") == "18:30" and h.requests() == []
    a = h.say("E101", "drop at 7:30 pm today", intent("change_time", "drop", TODAY, "19:30"))
    assert a["status"] == "applied" and h.slot("E101") == "19:30"


def test_rule_break_offers_next_option_and_employee_takes_it(h):
    # Acme allows moving at most 3 h; 22:00 is 3.5 h later. Nearest allowed slot is 21:30.
    r = h.say("E101", "drop at 10 pm today", intent("change_time", "drop", TODAY, "22:00"))
    assert r["status"] == "rule_violation" and "21:30" in r["reply"]
    r = h.say("E101", "ok, 21:30 then", intent("accept_offer", choice="alternative"))
    assert r["status"] == "applied" and h.slot("E101") == "21:30"


def test_offered_slot_named_with_wrong_am_pm_is_taken_as_the_offer(h):
    # Real models sometimes read "9:30 works" as 09:30 right after we offered 21:30.
    r = h.say("E101", "drop at 10 pm today", intent("change_time", "drop", TODAY, "22:00"))
    assert r["status"] == "rule_violation" and "21:30" in r["reply"]
    r = h.say("E101", "ok 9:30 works", intent("change_time", "drop", TODAY, "09:30"))
    assert r["status"] == "applied" and h.slot("E101") == "21:30"


def test_full_cab_offers_nearest_slot(h):
    r = h.say("E101", "8 pm drop today", intent("change_time", "drop", TODAY, "20:00"))
    assert r["status"] == "unavailable" and "full" in r["reply"]
    assert h.slot("E101") == "18:30"
    r = h.say("E101", "yes", intent("accept_offer"))
    assert r["status"] == "applied" and h.slot("E101") in ("19:30", "20:30")


def test_no_cab_at_requested_time(h):
    r = h.say("E101", "drop at 20:15", intent("change_time", "drop", TODAY, "20:15"))
    assert r["status"] == "unavailable" and "no drop cab at 20:15" in r["reply"]


def test_ambiguous_request_gets_questions_and_context_carries(h):
    # "a bit late" names no time and no day; the employee has drops today and tomorrow.
    r = h.say("E101", "Can I go a bit late?", intent("change_time", "drop"))
    assert r["status"] == "clarify" and "Which trip" in r["reply"]
    r = h.say("E101", "tonight", intent("change_time", date=TODAY))
    assert r["status"] == "clarify" and "What time" in r["reply"]
    # The follow-up only makes sense with the earlier messages: the prompt carries them.
    r = h.say("E101", "Make it 7 instead", intent("change_time", new_time="19:00"))
    assert '"direction": "drop"' in h.llm.prompts[-1] and TODAY in h.llm.prompts[-1]
    assert "Can I go a bit late?" in h.llm.prompts[-1]
    assert r["status"] == "applied" and h.slot("E101") == "19:00"
    assert h.slot("E101", TOMORROW) == "18:30"   # the other drop is untouched


def test_decline_clears_offer(h):
    h.say("E101", "8 pm drop today", intent("change_time", "drop", TODAY, "20:00"))
    assert h.say("E101", "no thanks", intent("decline"))["status"] == "info"
    r = h.say("E101", "yes", intent("accept_offer"))
    assert r["status"] == "clarify" and h.slot("E101") == "18:30"


def test_out_of_range_and_off_topic(h):
    r = h.say("E101", "change my drop on 25th", intent("change_time", "drop", "2026-11-25", "19:00"))
    assert r["status"] == "rejected" and "today and tomorrow" in r["reply"]
    assert h.say("E101", "tell me a joke", intent("other"))["status"] == "info"


def test_already_departed_trip_is_hard_rejected(h):
    at(18, 45)  # today's 18:30 drop has left; only tomorrow's trips remain upcoming
    r = h.say("E101", "drop today at 21:00", intent("change_time", "drop", TODAY, "21:00"))
    assert r["status"] == "clarify"   # no upcoming drop today -> asks which trip
    assert "tomorrow" in r["reply"]


# ---- tool selection --------------------------------------------------------------------
def test_tool_selection_for_each_kind_of_message(h):
    cases = [
        (intent("change_time", "drop", TODAY, "19:00"),
         ["get_upcoming_trips", "get_policy", "find_cab_options", "apply_change"]),
        (intent("status"), ["get_upcoming_trips", "get_pending_request"]),
        (intent("change_time", "drop"), ["get_upcoming_trips"]),          # clarify: no write tools
        (intent("other"), ["get_upcoming_trips"]),
    ]
    for parsed, expected in cases:
        h.tools.calls.clear()
        h.say("E102", "msg", parsed)
        assert h.tools.calls == expected, parsed


def test_rule_violation_never_calls_write_tools(h):
    at(15, 0)
    h.tools.calls.clear()
    h.say("G201", "drop 19:30", intent("change_time", "drop", TODAY, "19:30"))
    assert "apply_change" not in h.tools.calls and "create_approval_request" not in h.tools.calls


# ---- LLM budget ------------------------------------------------------------------------
def test_llm_calls_per_message_within_budget(h):
    r = h.say("E101", "drop 19:00 today", intent("change_time", "drop", TODAY, "19:00"))
    assert r["meta"]["llm_calls"] == 1
    r = h.say("E102", "drop 19:00 today", "not json at all", intent("change_time", "drop", TODAY, "19:00"))
    assert r["status"] == "applied" and r["meta"]["llm_calls"] == 2   # one repair retry
