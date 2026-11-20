"""Pure rule engine tests: no DB, no LLM."""
from datetime import datetime, timedelta

from app.clock import IST
from app.rules import Policy, check_change

ACME = Policy("acme", "Acme Corp", cutoff_minutes=120, max_shift_minutes=180, desk_phone="x")
GLOBEX = Policy("globex", "Globex India", cutoff_minutes=240, max_shift_minutes=120, desk_phone="y")
NOW = datetime(2026, 11, 19, 15, 0, tzinfo=IST)
DROP = NOW.replace(hour=18, minute=30)


def codes(v):
    return [x.code for x in v]


def test_same_request_different_company_rules():
    new = NOW.replace(hour=19, minute=30)
    assert check_change(ACME, NOW, DROP, new) == []                    # 3.5 h lead >= 2 h
    assert codes(check_change(GLOBEX, NOW, DROP, new)) == ["cutoff"]   # 3.5 h lead < 4 h


def test_cutoff_uses_earlier_of_current_and_new_time():
    # Moving to an EARLIER cab that leaves soon is also blocked.
    assert codes(check_change(ACME, NOW, DROP, NOW + timedelta(minutes=60))) == ["cutoff"]


def test_cutoff_boundary_is_inclusive():
    now = DROP - timedelta(minutes=120)
    assert check_change(ACME, now, DROP, DROP + timedelta(minutes=30)) == []
    assert codes(check_change(ACME, now + timedelta(minutes=1), DROP, DROP + timedelta(minutes=30))) == ["cutoff"]


def test_max_shift():
    assert codes(check_change(GLOBEX, NOW.replace(hour=9), DROP, DROP + timedelta(minutes=150))) == ["max_shift"]
    assert check_change(ACME, NOW.replace(hour=9), DROP, DROP + timedelta(minutes=150)) == []


def test_hard_violations_are_not_escalatable():
    v = check_change(ACME, NOW, NOW - timedelta(minutes=5), DROP)
    assert codes(v) == ["trip_started"] and not v[0].escalatable
    v = check_change(ACME, NOW, DROP, NOW - timedelta(minutes=5))
    assert codes(v) == ["in_past"] and not v[0].escalatable


def test_soft_violations_are_escalatable_and_explained():
    v = check_change(GLOBEX, NOW, DROP, DROP + timedelta(hours=3))
    assert set(codes(v)) == {"cutoff", "max_shift"}
    assert all(x.escalatable for x in v)
    assert "Globex India" in v[0].message
