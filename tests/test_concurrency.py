"""Bonus: last-seat race and double submits. Exactly one seat, exactly one request."""
from concurrent.futures import ThreadPoolExecutor

from tests.conftest import TODAY, intent

LAST_SEAT_CAB = "GLX-20261119-D-2000"   # Globex 20:00 drop has exactly one free seat


def test_two_employees_race_for_last_seat(h):
    h.auth("G201"), h.auth("G202")   # get tokens before racing
    want = intent("change_time", "drop", TODAY, "20:00")
    h.llm.push(want, want)

    def go(emp):
        return h.client.post("/chat", json={"message": "drop at 8 pm"}, headers=h.auth(emp)).json()

    with ThreadPoolExecutor(2) as ex:
        results = list(ex.map(go, ["G201", "G202"]))
    assert sorted(r["status"] for r in results) == ["applied", "unavailable"]
    cab = h.q("SELECT booked, capacity FROM cabs WHERE id = %s", LAST_SEAT_CAB)[0]
    assert cab["booked"] == cab["capacity"]
    assert len(h.requests()) == 1


def test_double_submit_same_change(h):
    h.auth("E101")
    want = intent("change_time", "drop", TODAY, "19:00")
    h.llm.push(want, want)

    def go(_):
        return h.client.post("/chat", json={"message": "drop at 7"}, headers=h.auth("E101")).json()

    with ThreadPoolExecutor(2) as ex:
        results = list(ex.map(go, range(2)))
    assert sorted(r["status"] for r in results) == ["applied", "no_change"]
    assert len(h.requests()) == 1
    assert h.q("SELECT booked FROM cabs WHERE id = 'ACM-20261119-D-1900'")[0]["booked"] == 3  # 2 others + 1


def test_idempotency_key_replays_response(h):
    h.auth("E101")
    h.llm.push(intent("change_time", "drop", TODAY, "19:00"))
    key = {"Idempotency-Key": "abc-123"}

    def go(_):
        return h.client.post("/chat", json={"message": "drop at 7"}, headers=h.auth("E101") | key).json()

    with ThreadPoolExecutor(2) as ex:
        a, b = list(ex.map(go, range(2)))
    assert a["request_id"] == b["request_id"] and {a.get("replayed"), b.get("replayed")} == {None, True}
    assert h.llm.calls == 1 and len(h.requests()) == 1
