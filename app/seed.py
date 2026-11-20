"""Sample data for two companies with different rules.

             cut-off   max move w/o approval   seats per cab
  Acme       2 h       3 h                     4
  Globex     4 h       2 h                     6

Cabs are generated for "today" and "tomorrow" (relative to app.clock.now()), so the
demo works on any day. `others` = riders not modelled as employees, used to make some
cabs full or nearly full.

Run:  python -m app.seed           (seed only if empty)
      python -m app.seed --reset   (wipe and reseed)
"""
import sys
from datetime import date, time, timedelta

from psycopg_pool import ConnectionPool

from app import clock
from app.db import init_schema, make_pool

COMPANIES = [
    # id, name, cutoff_min, max_shift_min, capacity, desk phone
    ("acme", "Acme Corp", 120, 180, 4, "+91-80-4000-1000"),
    ("globex", "Globex India", 240, 120, 6, "+91-80-4000-2000"),
]
PREFIX = {"acme": "ACM", "globex": "GLX"}
ROUTE = {"acme": "Whitefield", "globex": "Electronic City"}

EMPLOYEES = [
    ("E101", "acme", "Priya Sharma", "employee"),
    ("E102", "acme", "Rahul Verma", "employee"),
    ("E103", "acme", "Sana Khan", "employee"),
    ("M100", "acme", "Anil Mehta (Transport Manager)", "manager"),
    ("G201", "globex", "Arjun Rao", "employee"),
    ("G202", "globex", "Meera Iyer", "employee"),
    ("G203", "globex", "Kiran Das", "employee"),
    ("M200", "globex", "Deepa Nair (Transport Manager)", "manager"),
]

DROP_SLOTS = [time(h, m) for h in range(17, 23) for m in (0, 30)] + [time(23, 0)]
PICKUP_SLOTS = [time(8, 0), time(9, 0), time(10, 0)]

# Riders not modelled as employees, per company and drop slot (same pattern both days).
OTHERS = {
    "acme": {time(19, 0): 2, time(20, 0): 4, time(20, 30): 3, time(21, 0): 1},      # 20:00 full, 20:30 one seat
    "globex": {time(19, 0): 3, time(20, 0): 5, time(21, 0): 6},                     # 20:00 one seat, 21:00 full
}

# Every employee: drop today 18:30, drop tomorrow 18:30, pickup tomorrow 09:00.
ROSTER = [(0, "drop", time(18, 30)), (1, "drop", time(18, 30)), (1, "pickup", time(9, 0))]


def cab_id(company: str, d: date, direction: str, slot: time) -> str:
    return f"{PREFIX[company]}-{d:%Y%m%d}-{'D' if direction == 'drop' else 'P'}-{slot:%H%M}"


def seed(pool: ConnectionPool, today: date) -> None:
    with pool.connection() as conn, conn.transaction():
        cur = conn.cursor()
        for c in COMPANIES:
            cur.execute("INSERT INTO companies VALUES (%s,%s,%s,%s,%s,%s)", c)
        for emp_id, company, name, role in EMPLOYEES:
            cur.execute(
                "INSERT INTO employees (id, company_id, name, role, route) VALUES (%s,%s,%s,%s,%s)",
                (emp_id, company, name, role, ROUTE[company] if role == "employee" else None),
            )
        for company, _, _, _, capacity, _ in COMPANIES:
            for offset in (0, 1):
                d = today + timedelta(days=offset)
                for direction, slots in (("drop", DROP_SLOTS), ("pickup", PICKUP_SLOTS)):
                    for slot in slots:
                        others = OTHERS[company].get(slot, 0) if direction == "drop" else 1
                        cur.execute(
                            "INSERT INTO cabs (id, company_id, route, trip_date, direction, slot, capacity, booked)"
                            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                            (cab_id(company, d, direction, slot), company, ROUTE[company], d, direction,
                             slot, capacity, others),
                        )
        for emp_id, company, _, role in EMPLOYEES:
            if role != "employee":
                continue
            for offset, direction, slot in ROSTER:
                d = today + timedelta(days=offset)
                cid = cab_id(company, d, direction, slot)
                cur.execute(
                    "INSERT INTO roster (company_id, employee_id, trip_date, direction, cab_id) VALUES (%s,%s,%s,%s,%s)",
                    (company, emp_id, d, direction, cid),
                )
                cur.execute("UPDATE cabs SET booked = booked + 1 WHERE id = %s", (cid,))


def reset(pool: ConnectionPool, today: date) -> None:
    """Drop everything (including LangGraph checkpoints) and reseed."""
    with pool.connection() as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    init_schema(pool)
    seed(pool, today)


def seed_if_empty(pool: ConnectionPool, today: date) -> bool:
    init_schema(pool)
    with pool.connection() as conn:
        if conn.execute("SELECT count(*) AS n FROM companies").fetchone()["n"]:
            return False
    seed(pool, today)
    return True


if __name__ == "__main__":
    p = make_pool()
    if "--reset" in sys.argv:
        reset(p, clock.now().date())
        print("Database reset and seeded.")
    else:
        print("Seeded." if seed_if_empty(p, clock.now().date()) else "Data already present, not reseeding.")
    p.close()
