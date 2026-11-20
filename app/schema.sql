-- Every business table carries company_id. Every query in app/tools.py filters on it.

CREATE TABLE IF NOT EXISTS companies (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    cutoff_minutes    INT  NOT NULL,   -- change must arrive this long before the trip
    max_shift_minutes INT  NOT NULL,   -- largest move allowed without manager approval
    cab_capacity      INT  NOT NULL,   -- seats per cab for this company
    desk_phone        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS employees (
    id         TEXT PRIMARY KEY,
    company_id TEXT NOT NULL REFERENCES companies(id),
    name       TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('employee', 'manager')),
    route      TEXT
);

CREATE TABLE IF NOT EXISTS cabs (
    id         TEXT PRIMARY KEY,
    company_id TEXT NOT NULL REFERENCES companies(id),
    route      TEXT NOT NULL,
    trip_date  DATE NOT NULL,
    direction  TEXT NOT NULL CHECK (direction IN ('pickup', 'drop')),
    slot       TIME NOT NULL,
    capacity   INT  NOT NULL,
    booked     INT  NOT NULL DEFAULT 0,
    CHECK (booked >= 0 AND booked <= capacity)   -- last line of defence against overbooking
);

CREATE TABLE IF NOT EXISTS roster (
    id          SERIAL PRIMARY KEY,
    company_id  TEXT NOT NULL REFERENCES companies(id),
    employee_id TEXT NOT NULL REFERENCES employees(id),
    trip_date   DATE NOT NULL,
    direction   TEXT NOT NULL CHECK (direction IN ('pickup', 'drop')),
    cab_id      TEXT NOT NULL REFERENCES cabs(id),
    UNIQUE (employee_id, trip_date, direction)
);

CREATE TABLE IF NOT EXISTS shift_requests (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id  TEXT NOT NULL REFERENCES companies(id),
    employee_id TEXT NOT NULL REFERENCES employees(id),
    roster_id   INT  NOT NULL REFERENCES roster(id),
    from_cab_id TEXT NOT NULL REFERENCES cabs(id),
    to_cab_id   TEXT NOT NULL REFERENCES cabs(id),
    status      TEXT NOT NULL CHECK (status IN ('applied', 'pending_approval', 'rejected', 'failed')),
    violations  JSONB NOT NULL DEFAULT '[]',
    thread_id   TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_by  TEXT,
    decided_at  TIMESTAMPTZ,
    note        TEXT
);
-- At most one open approval per trip, even if the employee double-submits.
CREATE UNIQUE INDEX IF NOT EXISTS one_pending_per_trip
    ON shift_requests (roster_id) WHERE status = 'pending_approval';

CREATE TABLE IF NOT EXISTS notifications (
    id          SERIAL PRIMARY KEY,
    company_id  TEXT NOT NULL REFERENCES companies(id),
    employee_id TEXT NOT NULL REFERENCES employees(id),
    text        TEXT NOT NULL,
    seen        BOOLEAN NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_idempotency (
    company_id  TEXT NOT NULL,
    employee_id TEXT NOT NULL,
    key         TEXT NOT NULL,
    response    JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (company_id, employee_id, key)
);

-- Who asked for what, and who decided.
CREATE TABLE IF NOT EXISTS audit_log (
    id         BIGSERIAL PRIMARY KEY,
    company_id TEXT NOT NULL,
    actor      TEXT NOT NULL,
    event      TEXT NOT NULL,
    details    JSONB NOT NULL DEFAULT '{}',
    at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
