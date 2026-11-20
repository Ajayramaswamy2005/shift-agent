"""Company rules as plain functions. No LLM, no database: easy to test and to reason about.

The agent may only change a trip if check_change() returns no violations.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Policy:
    company_id: str
    company_name: str
    cutoff_minutes: int
    max_shift_minutes: int
    desk_phone: str


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    escalatable: bool  # can a transport manager override it?


def _fmt_duration(td: timedelta) -> str:
    minutes = max(0, int(td.total_seconds() // 60))
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h} h {m} min"
    return f"{h} h" if h else f"{m} min"


def check_change(policy: Policy, now: datetime, current: datetime, new: datetime) -> list[Violation]:
    """Return every rule the change would break (empty list = allowed).

    Assumption: the cut-off applies to the earlier of the two times, so you can neither
    change a trip that is about to leave nor jump onto a cab that is about to leave.
    """
    if current <= now:
        return [Violation("trip_started", "this trip has already left", escalatable=False)]
    if new <= now:
        return [Violation("in_past", f"{new:%H:%M} has already passed", escalatable=False)]

    violations = []
    lead = min(current, new) - now
    if lead < timedelta(minutes=policy.cutoff_minutes):
        violations.append(Violation(
            "cutoff",
            f"{policy.company_name} needs changes at least {_fmt_duration(timedelta(minutes=policy.cutoff_minutes))} "
            f"before the trip, and this one is only {_fmt_duration(lead)} away",
            escalatable=True,
        ))
    shift = abs(new - current)
    if shift > timedelta(minutes=policy.max_shift_minutes):
        violations.append(Violation(
            "max_shift",
            f"{policy.company_name} allows moving a trip by at most "
            f"{_fmt_duration(timedelta(minutes=policy.max_shift_minutes))} without approval, "
            f"and this moves it by {_fmt_duration(shift)}",
            escalatable=True,
        ))
    return violations
