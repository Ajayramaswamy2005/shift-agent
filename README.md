# Shift-Change Request Agent

MoveInSync campus case study A. A chat agent that lets an employee change their cab pickup or
drop time in plain language. It reads only that employee's roster, applies their company's rules
in code, books the seat atomically, or explains why not and offers the next option. When a rule
is broken and the employee still wants the change, it escalates to the transport manager, pauses,
and resumes when the decision arrives, even after a restart.

**Demo video:** _<add your 2-minute recording link here>_

## Run it (one command)

```bash
cp .env.example .env          # add a free Groq key as LLM_API_KEY (optional for tests)
docker compose up --build     # app on http://localhost:8000 (demo page at /)
```

Tests and the evaluation set (no API key needed, they use a fake LLM):

```bash
docker compose run --rm tests          # pytest (44 tests) + eval/run_eval.py -> eval/eval.jsonl
```

Restart test by hand: escalate a request in the demo page, run `docker compose restart app`, then
approve it from the manager panel. The employee's conversation resumes.

Without Docker: Postgres 16 running locally, then `pip install -r requirements.txt`,
`python -m app.seed`, `uvicorn app.main:app`, `pytest`.

## Sample data

Two companies, generated for "today" and "tomorrow" so the demo works on any day:

|                     | Acme Corp (route Whitefield) | Globex India (route Electronic City) |
|---------------------|------------------------------|--------------------------------------|
| Cut-off             | 2 h before the trip          | 4 h before the trip                  |
| Max move w/o approval | 3 h                        | 2 h                                  |
| Seats per cab       | 4                            | 6                                    |
| Employees           | E101, E102, E103             | G201, G202, G203                     |
| Manager             | M100                         | M200                                 |
| Notable cabs (drop) | 20:00 full, 20:30 one seat   | 20:00 one seat, 21:00 full           |

Every employee has a drop today at 18:30, a pickup tomorrow at 09:00 and a drop tomorrow at 18:30.
`DEMO_NOW=2026-11-19T15:00` in `.env` freezes the clock for a predictable recording.

## Architecture

```
 POST /chat (JWT: company + employee)                       POST /manager/requests/{id}/decision (JWT: manager)
        |                                                                 |
  per-employee Postgres advisory lock, optional Idempotency-Key           | claim request (company-scoped)
        |                                                                 |
  LangGraph  (state checkpointed in Postgres, thread = company:employee)  |
  ┌────────────┐   ┌─────────┐   ┌──────────┐   ┌───────┐                 |
  │ understand │──▶│ resolve │──▶│ evaluate │──▶│ apply │──▶ finish       |
  │  (LLM ×1)  │   │ (code)  │   │ (rules + │   └───────┘                 |
  └────────────┘   └────┬────┘   │  seats)  │                             |
                        │        └──────────┘                             ▼
                        └──▶ escalate ──▶ wait_for_manager (interrupt) ──▶ after_decision ──▶ finish
```

| File | What it does |
|---|---|
| `app/rules.py` | Company rules as pure functions (cut-off, max move). No LLM, no DB. |
| `app/tools.py` | The agent's tools: read roster, check cab options, apply change, raise approval request. Every query filters by the token's company. |
| `app/llm.py` | One job: message → validated JSON intent. Strict whitelist validation. Groq/Gemini/any OpenAI-compatible API; `ScriptedLLM` for tests. |
| `app/agent.py` | The LangGraph graph, reply templates, LLM call budget. |
| `app/main.py` | FastAPI endpoints, JWT checks, locking, idempotency, error handling. |
| `app/auth.py` | JWT issue/verify. `/dev/token` is the mock issuer. |
| `eval/` | `cases.json` (25 scenarios) → `run_eval.py` → `eval.jsonl`. |

## Design decisions

**1. The LLM only understands; code decides and replies.** The model turns the message (plus the
last few turns and what we already know) into `{intent, direction, date, new_time, choice}`. Picking
the trip, checking rules, finding seats, booking, escalating and writing the reply are all plain
Python. Replies are templates filled from facts, so the agent cannot announce a change that did not
happen. The case study warns that "a confident made-up confirmation is worse than no answer".
Understanding "make it 8 instead" in context is where an LLM earns its place. Deciding whether 3 h
30 min is less than 4 h is not.

**2. Identity only from the token.** `company_id` and `employee_id` come from the verified JWT and
are written into graph state by the server on every call. The conversation thread id is derived
from them. The chat body accepts only `message` (`extra="forbid"`, so a `company_id` field gets a
422). Model output passes through a whitelist (`validate_intent`): unknown keys such as
`company_id` or `approved: true` are dropped. Every SQL query filters by the token's company.
Managers can only see and decide their own company's requests, and anything else returns 404, not
403, so another company's request ids can't even be confirmed to exist.

**3. Rules in code, two kinds of violation.** `trip_started` and `in_past` are hard: no one can
override them. `cutoff` and `max_shift` are soft: the agent explains, offers the nearest slot that
breaks no rule, and offers to ask the manager. A broken rule never leads to a write; there is a test
for exactly that.

**4. Approval survives restarts.** The escalate node writes a `pending_approval` row (with the
thread id), then `wait_for_manager` calls LangGraph `interrupt()`. The checkpoint lives in Postgres
(`PostgresSaver`), so nothing is in memory. The manager endpoint finds the thread from the DB row and
resumes it with `Command(resume=decision)`. Before applying, it re-checks the seat and the time,
because the cab may have filled during the wait. `claim_request` (a conditional UPDATE) stops two
managers deciding the same request twice.

**5. Seats and double submits (bonus).** Booking a seat is
`UPDATE cabs SET booked = booked + 1 WHERE … AND booked < capacity` inside one transaction that
also locks the roster row (`FOR UPDATE`). A `CHECK (booked <= capacity)` constraint is the last line
of defence. A partial unique index allows at most one pending approval per trip. Each employee's
messages are serialised with a Postgres advisory lock, and an optional `Idempotency-Key` header
replays the first response.

**6. Failures become plain sentences.** DB queries run under a 2 s `statement_timeout`, and the
LLM has a 4 s timeout. Tool errors, LLM errors and a cab filling mid-conversation each have their
own message saying what happened, that nothing was changed, and what to do next (try again, pick
another slot, or call the desk number). A global handler logs any unexpected error server-side and
returns a generic message. Stack traces never reach the user.

## How it stays under 4 LLM calls and 5 s p95

- Only the `understand` node calls the LLM: **1 call** normally, **2** if the first output fails
  validation (one repair attempt with the error message). Replies are templates, so 0 calls there.
- `LLMBudget` enforces a hard cap of 4 per message in code, and the API returns `meta.llm_calls` for
  each message.
- Messages sent while a request is waiting for the manager use **0 calls**: the status reply is built
  from the database.
- Latency: one small-model call (Groq `openai/gpt-oss-20b` with `reasoning_effort=low`, JSON mode) plus a
  few indexed queries. Worst case is bounded by timeouts: 2 × 4 s for the LLM, 2 s per query.
  Measured numbers are in `EVALUATION.md`.

## Trade-offs

- **Template replies vs LLM-written replies.** Templates are less warm and less varied, but they are
  always true, cost no LLM call, and halve latency. For a transport desk, correct beats chatty.
- **A fixed graph instead of a free tool-calling agent.** A ReAct-style loop would be more flexible,
  but the number of LLM calls and the tool order would be unpredictable, and the 4-call limit could
  only be hoped for. The fixed graph makes the tool order testable (see `tools-01..03` in the eval).
- **Postgres for everything (data, checkpoints, locks, idempotency).** One dependency instead of
  Postgres plus Redis. Redis would be faster for locks, but this load doesn't need it, and the
  transactional guarantees matter more.
- **The cut-off is applied to the earlier of the current and the new time.** This is an assumption: it
  stops both changing a trip that is about to leave and jumping onto a cab that is about to leave.

## Known limitations

- Only today's and tomorrow's trips. One conversation thread per employee; its message history grows
  without trimming.
- A pending request does not expire on its own. If the manager approves after the trip time, the
  employee is told it came too late.
- If the process is killed in the middle of a resume, the request can stay claimed and need a manual
  reset. Ordinary exceptions un-claim it automatically.
- AM/PM for bare numbers ("make it 8") is left to the model, with the hint that evening drops are PM.
  The confirmation always states the exact 24-hour time, so a misread is visible at once, but it is
  already booked by then. A confirm-before-apply step would fix this at the cost of one more turn.
- Seat counts include "other riders" who aren't modelled as employees (to create full cabs).
- The mock JWT issuer (`/dev/token`) must be disabled in production (`DEV_TOKEN_ISSUER=0`).

## API

| Method | Path | Who | Notes |
|---|---|---|---|
| POST | `/dev/token` | anyone (demo) | `{"user_id": "E101"}` → JWT. Company and role come from the DB, not the request. |
| POST | `/chat` | employee | `{"message": "..."}`, optional `Idempotency-Key` header. Returns `status`, `reply`, `request_id`, `updates`, `meta.llm_calls`. |
| GET | `/chat/history` | employee | Conversation from the checkpoint. |
| GET | `/manager/requests` | manager | Pending approvals for the manager's company. |
| POST | `/manager/requests/{id}/decision` | manager | `{"approved": true, "note": "..."}`. Resumes the paused conversation. |

`status` values: `applied`, `no_change`, `clarify`, `rule_violation`, `unavailable`, `rejected`,
`pending_approval`, `info`, `error`, `busy`.
