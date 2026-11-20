"""The agent's tools: mock transport APIs over Postgres.

Isolation rule: every function takes an Identity built from the JWT and filters every
query by company_id (and employee_id where it applies). Nothing here accepts a company
or employee id from the model.

Trips and cabs are returned as plain dicts so they can live in LangGraph state.
"""
import json
from datetime import date, datetime, time
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool

from app.auth import Identity
from app.clock import IST
from app.rules import Policy


class ToolError(Exception):
    """A tool could not do its job (database down, bad data...)."""


class ToolTimeout(ToolError):
    """A tool took too long."""


class CabFull(Exception):
    """The target cab has no free seat (may have filled while the employee was deciding)."""


class TripChanged(Exception):
    """The roster no longer matches what the agent saw (changed by another request)."""


def trip_datetime(trip: dict) -> datetime:
    return datetime.combine(date.fromisoformat(trip["date"]), time.fromisoformat(trip["slot"]), IST)


def _trip(row: dict) -> dict:
    return {
        "roster_id": row["roster_id"],
        "date": row["trip_date"].isoformat(),
        "direction": row["direction"],
        "cab_id": row["cab_id"],
        "slot": row["slot"].strftime("%H:%M"),
        "route": row["route"],
    }


class TransportTools:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        self.calls: list[str] = []          # which tools ran, for tool-selection tests
        self.faults: dict[str, str] = {}    # fault injection: {"get_upcoming_trips": "timeout" | "error"}

    # -- plumbing -----------------------------------------------------------------
    def _enter(self, name: str) -> None:
        self.calls.append(name)
        fault = self.faults.get(name)
        if fault == "timeout":
            raise ToolTimeout(f"{name} timed out")
        if fault == "error":
            raise ToolError(f"{name} failed")

    def _run(self, fn):
        try:
            return fn()
        except psycopg.errors.QueryCanceled as e:      # statement_timeout hit
            raise ToolTimeout(str(e)) from e
        except psycopg.OperationalError as e:
            raise ToolError(str(e)) from e

    @staticmethod
    def _audit(cur, company_id: str, actor: str, event: str, details: dict[str, Any]) -> None:
        cur.execute(
            "INSERT INTO audit_log (company_id, actor, event, details) VALUES (%s,%s,%s,%s)",
            (company_id, actor, event, json.dumps(details, default=str)),
        )

    # -- read tools ---------------------------------------------------------------
    def get_policy(self, ident: Identity) -> Policy:
        self._enter("get_policy")

        def q():
            with self.pool.connection() as conn:
                r = conn.execute("SELECT * FROM companies WHERE id = %s", (ident.company_id,)).fetchone()
            if not r:
                raise ToolError("company not found")
            return Policy(r["id"], r["name"], r["cutoff_minutes"], r["max_shift_minutes"], r["desk_phone"])
        return self._run(q)

    def get_upcoming_trips(self, ident: Identity, now: datetime) -> list[dict]:
        """Read the roster: this employee's trips that have not left yet (today and tomorrow)."""
        self._enter("get_upcoming_trips")

        def q():
            with self.pool.connection() as conn:
                rows = conn.execute(
                    """SELECT r.id AS roster_id, r.trip_date, r.direction, r.cab_id, c.slot, c.route
                       FROM roster r JOIN cabs c ON c.id = r.cab_id AND c.company_id = r.company_id
                       WHERE r.company_id = %s AND r.employee_id = %s
                         AND (r.trip_date > %s OR (r.trip_date = %s AND c.slot > %s))
                       ORDER BY r.trip_date, c.slot""",
                    (ident.company_id, ident.user_id, now.date(), now.date(), now.time()),
                ).fetchall()
            return [_trip(r) for r in rows]
        return self._run(q)

    def find_cab_options(self, ident: Identity, trip: dict) -> list[dict]:
        """Check cab options: other cabs on the same route, day and direction, with free seats."""
        self._enter("find_cab_options")

        def q():
            with self.pool.connection() as conn:
                rows = conn.execute(
                    """SELECT id, slot, capacity, booked FROM cabs
                       WHERE company_id = %s AND route = %s AND trip_date = %s AND direction = %s
                       ORDER BY slot""",
                    (ident.company_id, trip["route"], trip["date"], trip["direction"]),
                ).fetchall()
            return [{"cab_id": r["id"], "slot": r["slot"].strftime("%H:%M"),
                     "seats_free": r["capacity"] - r["booked"]} for r in rows]
        return self._run(q)

    def get_pending_request(self, ident: Identity) -> dict | None:
        self._enter("get_pending_request")

        def q():
            with self.pool.connection() as conn:
                return conn.execute(
                    """SELECT s.id, s.created_at, s.to_cab_id, c.slot AS to_slot
                       FROM shift_requests s JOIN cabs c ON c.id = s.to_cab_id
                       WHERE s.company_id = %s AND s.employee_id = %s AND s.status = 'pending_approval'
                       ORDER BY s.created_at DESC LIMIT 1""",
                    (ident.company_id, ident.user_id),
                ).fetchone()
        return self._run(q)

    # -- write tools --------------------------------------------------------------
    def apply_change(self, ident: Identity, trip: dict, to_cab_id: str, thread_id: str,
                     request_id: str | None = None, decided_by: str | None = None) -> dict:
        """Raise a shift-change request and apply it, atomically.

        Returns {"request_id", "already_done"}. Raises CabFull or TripChanged.
        The seat is taken with a conditional UPDATE (booked < capacity) inside one
        transaction, so two people can never get the same last seat.
        """
        self._enter("apply_change")

        def tx():
            with self.pool.connection() as conn, conn.transaction():
                cur = conn.cursor()
                row = cur.execute(
                    "SELECT id, cab_id FROM roster WHERE id = %s AND company_id = %s AND employee_id = %s FOR UPDATE",
                    (trip["roster_id"], ident.company_id, ident.user_id),
                ).fetchone()
                if row is None:
                    raise ToolError("trip not found for this employee")
                if row["cab_id"] == to_cab_id:          # double submit: already done, create nothing
                    return {"request_id": request_id, "already_done": True}
                if row["cab_id"] != trip["cab_id"]:
                    raise TripChanged()
                target = cur.execute(
                    """UPDATE cabs SET booked = booked + 1
                       WHERE id = %s AND company_id = %s AND route = %s AND trip_date = %s
                         AND direction = %s AND booked < capacity
                       RETURNING id""",
                    (to_cab_id, ident.company_id, trip["route"], trip["date"], trip["direction"]),
                ).fetchone()
                if target is None:
                    raise CabFull()
                cur.execute("UPDATE cabs SET booked = booked - 1 WHERE id = %s AND company_id = %s AND booked > 0",
                            (trip["cab_id"], ident.company_id))
                cur.execute("UPDATE roster SET cab_id = %s WHERE id = %s AND company_id = %s",
                            (to_cab_id, trip["roster_id"], ident.company_id))
                if request_id:
                    cur.execute(
                        """UPDATE shift_requests SET status = 'applied', decided_at = now(),
                           decided_by = COALESCE(decided_by, %s) WHERE id = %s AND company_id = %s""",
                        (decided_by, request_id, ident.company_id),
                    )
                    rid = request_id
                else:
                    rid = str(cur.execute(
                        """INSERT INTO shift_requests (company_id, employee_id, roster_id, from_cab_id, to_cab_id,
                                                       status, thread_id)
                           VALUES (%s,%s,%s,%s,%s,'applied',%s) RETURNING id""",
                        (ident.company_id, ident.user_id, trip["roster_id"], trip["cab_id"], to_cab_id, thread_id),
                    ).fetchone()["id"])
                self._audit(cur, ident.company_id, decided_by or ident.user_id, "change_applied",
                            {"request_id": rid, "employee": ident.user_id, "from": trip["cab_id"], "to": to_cab_id})
                return {"request_id": rid, "already_done": False}
        return self._run(tx)

    def create_approval_request(self, ident: Identity, trip: dict, to_cab_id: str,
                                violations: list[dict], thread_id: str) -> str:
        """Raise a shift-change request that needs the transport manager. Idempotent per trip."""
        self._enter("create_approval_request")

        def tx():
            with self.pool.connection() as conn, conn.transaction():
                cur = conn.cursor()
                row = cur.execute(
                    """INSERT INTO shift_requests (company_id, employee_id, roster_id, from_cab_id, to_cab_id,
                                                   status, violations, thread_id)
                       VALUES (%s,%s,%s,%s,%s,'pending_approval',%s,%s)
                       ON CONFLICT (roster_id) WHERE status = 'pending_approval' DO NOTHING
                       RETURNING id""",
                    (ident.company_id, ident.user_id, trip["roster_id"], trip["cab_id"], to_cab_id,
                     json.dumps(violations), thread_id),
                ).fetchone()
                if row is None:
                    row = cur.execute(
                        "SELECT id FROM shift_requests WHERE roster_id = %s AND company_id = %s AND status = 'pending_approval'",
                        (trip["roster_id"], ident.company_id),
                    ).fetchone()
                rid = str(row["id"])
                self._audit(cur, ident.company_id, ident.user_id, "approval_requested",
                            {"request_id": rid, "to": to_cab_id, "violations": violations})
                return rid
        return self._run(tx)

    # -- manager side (scoped by the manager's company) --------------------------------
    def list_pending(self, company_id: str) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.execute(
                """SELECT s.id, s.employee_id, e.name AS employee_name, s.from_cab_id, s.to_cab_id,
                          s.violations, s.created_at
                   FROM shift_requests s JOIN employees e ON e.id = s.employee_id
                   WHERE s.company_id = %s AND s.status = 'pending_approval' ORDER BY s.created_at""",
                (company_id,),
            ).fetchall()

    def get_request(self, company_id: str, request_id: str) -> dict | None:
        with self.pool.connection() as conn:
            return conn.execute(
                "SELECT * FROM shift_requests WHERE id = %s AND company_id = %s", (request_id, company_id)
            ).fetchone()

    def claim_request(self, company_id: str, request_id: str, manager_id: str) -> dict | None:
        """Mark a pending request as being decided, so two decisions can't both run."""
        with self.pool.connection() as conn:
            return conn.execute(
                """UPDATE shift_requests SET decided_by = %s
                   WHERE id = %s AND company_id = %s AND status = 'pending_approval' AND decided_by IS NULL
                   RETURNING *""",
                (manager_id, request_id, company_id),
            ).fetchone()

    def unclaim_request(self, company_id: str, request_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE shift_requests SET decided_by = NULL WHERE id = %s AND company_id = %s AND status = 'pending_approval'",
                (request_id, company_id),
            )

    def close_request(self, company_id: str, request_id: str, status: str, actor: str, note: str) -> None:
        with self.pool.connection() as conn, conn.transaction():
            cur = conn.cursor()
            cur.execute(
                """UPDATE shift_requests SET status = %s, note = %s, decided_at = now(),
                   decided_by = COALESCE(decided_by, %s) WHERE id = %s AND company_id = %s""",
                (status, note, actor, request_id, company_id),
            )
            self._audit(cur, company_id, actor, f"request_{status}", {"request_id": request_id, "note": note})

    # -- notifications --------------------------------------------------------------
    def notify(self, company_id: str, employee_id: str, text: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("INSERT INTO notifications (company_id, employee_id, text) VALUES (%s,%s,%s)",
                         (company_id, employee_id, text))

    def pop_notifications(self, ident: Identity) -> list[str]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """UPDATE notifications SET seen = true
                   WHERE company_id = %s AND employee_id = %s AND NOT seen RETURNING id, text""",
                (ident.company_id, ident.user_id),
            ).fetchall()
        return [r["text"] for r in sorted(rows, key=lambda r: r["id"])]
