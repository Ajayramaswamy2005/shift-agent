"""One place to ask "what time is it?", so tests and demos can freeze time."""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
_frozen: datetime | None = None


def now() -> datetime:
    if _frozen is not None:
        return _frozen
    demo = os.getenv("DEMO_NOW")  # e.g. 2026-11-19T14:00
    if demo:
        dt = datetime.fromisoformat(demo)
        return dt if dt.tzinfo else dt.replace(tzinfo=IST)
    return datetime.now(IST)


def freeze(dt: datetime | None) -> None:
    global _frozen
    _frozen = dt if (dt is None or dt.tzinfo) else dt.replace(tzinfo=IST)
