"""Test harness. Uses a real Postgres (separate test database) and a scripted fake LLM.

Frozen clock: Thursday 2026-11-19 14:00 IST. Seeded facts the tests rely on:
  Acme   (cut-off 2 h, max move 3 h, 4 seats): drop 18:30; 19:00 free; 19:30 free; 20:00 FULL; 20:30 one seat
  Globex (cut-off 4 h, max move 2 h, 6 seats): drop 18:30; 19:30 free; 20:00 ONE seat; 21:00 FULL
"""
import os
from datetime import datetime

os.environ.setdefault("DATABASE_URL", "postgresql://postgres@localhost:5432/shift_test")

import psycopg  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import clock  # noqa: E402
from app.db import make_pool  # noqa: E402
from app.llm import ScriptedLLM  # noqa: E402
from app.main import create_app  # noqa: E402
from app.seed import reset  # noqa: E402

NOW = datetime(2026, 11, 19, 14, 0, tzinfo=clock.IST)
TODAY, TOMORROW = "2026-11-19", "2026-11-20"


def intent(intent="change_time", direction=None, date=None, new_time=None, choice=None, **extra):
    return {"intent": intent, "direction": direction, "date": date, "new_time": new_time, "choice": choice, **extra}


def _ensure_test_db() -> None:
    url = os.environ["DATABASE_URL"]
    name = url.rsplit("/", 1)[1]
    admin = url.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            conn.execute(f'CREATE DATABASE "{name}"')


class Harness:
    def __init__(self):
        self.llm = ScriptedLLM()
        self.client: TestClient | None = None
        self._tokens: dict[str, str] = {}
        self.start()

    # app lifecycle --------------------------------------------------------------
    def start(self):
        self.client = TestClient(create_app(llm=self.llm, seed=False))
        self.client.__enter__()

    def stop(self):
        self.client.__exit__(None, None, None)

    def restart(self):
        """Simulate a service restart: new app, new DB pool, new graph object."""
        self.stop()
        self.start()

    @property
    def tools(self):
        return self.client.app.state.tools

    # helpers ----------------------------------------------------------------------
    def auth(self, user_id: str) -> dict:
        if user_id not in self._tokens:
            r = self.client.post("/dev/token", json={"user_id": user_id})
            assert r.status_code == 200, r.text
            self._tokens[user_id] = r.json()["token"]
        return {"Authorization": f"Bearer {self._tokens[user_id]}"}

    def say(self, user_id: str, message: str, *llm_responses, headers: dict | None = None) -> dict:
        self.llm.push(*llm_responses)
        r = self.client.post("/chat", json={"message": message}, headers=self.auth(user_id) | (headers or {}))
        assert r.status_code == 200, r.text
        return r.json()

    def decide(self, manager_id: str, request_id: str, approved: bool, note: str | None = None):
        return self.client.post(f"/manager/requests/{request_id}/decision",
                                json={"approved": approved, "note": note}, headers=self.auth(manager_id))

    def q(self, sql: str, *params):
        with self.client.app.state.pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    def slot(self, employee_id: str, date: str = TODAY, direction: str = "drop") -> str:
        rows = self.q("SELECT c.slot FROM roster r JOIN cabs c ON c.id = r.cab_id "
                      "WHERE r.employee_id = %s AND r.trip_date = %s AND r.direction = %s",
                      employee_id, date, direction)
        return rows[0]["slot"].strftime("%H:%M")

    def requests(self, status: str | None = None) -> list[dict]:
        if status:
            return self.q("SELECT * FROM shift_requests WHERE status = %s", status)
        return self.q("SELECT * FROM shift_requests")


@pytest.fixture(scope="session", autouse=True)
def _test_db():
    _ensure_test_db()


@pytest.fixture
def h():
    clock.freeze(NOW)
    pool = make_pool()
    reset(pool, NOW.date())
    pool.close()
    harness = Harness()
    yield harness
    harness.stop()
    clock.freeze(None)


def at(hh: int, mm: int = 0):
    clock.freeze(NOW.replace(hour=hh, minute=mm))
