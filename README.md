# Shift-Change Request Agent

## Requirements

- Docker Desktop (with Docker Compose)
- A free Groq API key from https://console.groq.com/keys (only needed for the live chat and the real-model eval)

## 1. Configure

```bash
cp .env.example .env
```

Open `.env` and set `LLM_API_KEY=` to your Groq key. Change `JWT_SECRET` to any long random string.

## 2. Run the app

```bash
docker compose up --build
```

Open http://localhost:8000. Pick an employee on the left to chat, and a manager on the right to approve or reject requests.

To reset the demo data:

```bash
docker compose exec app python -m app.seed --reset
docker compose restart app
```

## 3. Run the tests and the evaluation set

No API key needed (uses a scripted fake LLM):

```bash
docker compose run --rm tests
```

This runs pytest and then `eval/run_eval.py`, which writes `eval/eval.jsonl` and `eval/summary.json`.

## 4. Run the evaluation with the real model (optional)

```bash
docker compose run --rm -e LLM_API_KEY=<your key> -e LLM_MODEL=openai/gpt-oss-20b \
  -e LLM_REASONING_EFFORT=low -e LLM_MAX_TOKENS=300 -e LLM_TIMEOUT_S=4.0 -e EVAL_PACE_S=7 \
  -v "$(pwd)/eval:/srv/eval" tests python -m eval.run_eval --llm real
```

Results go to `eval/eval_real_llm.jsonl` and `eval/summary_real_llm.json`. `EVAL_PACE_S=7` keeps the run under the Groq free tier's tokens-per-minute limit.

Held-out phrasings against the running app (needs step 2 running and Python with `httpx` installed):

```bash
python eval/heldout.py
```

## Without Docker

Needs Python 3.13 and PostgreSQL 16 running locally (`postgresql://postgres:postgres@localhost:5432/shift`, or set `DATABASE_URL`).

```bash
pip install -r requirements.txt
python -m app.seed
uvicorn app.main:app
pytest
```

## Design decisions

```
 POST /chat (JWT: company + employee)                     POST /manager/requests/{id}/decision (JWT: manager)
        |                                                               |
  per-employee advisory lock, optional Idempotency-Key                  | claim request (company-scoped)
        |                                                               |
  LangGraph (state checkpointed in Postgres, thread = company:employee) |
  understand ──▶ resolve ──▶ evaluate ──▶ apply ──▶ finish              |
   (LLM ×1)      (code)      (rules+seats)                              ▼
                    └──▶ escalate ──▶ wait_for_manager (interrupt) ──▶ after_decision ──▶ finish
```

1. **The LLM only understands; code decides and replies.** The model turns the message (plus recent
   turns) into `{intent, direction, date, new_time, choice}`. Picking the trip, checking rules, finding
   seats, booking, escalating and writing the reply are plain Python. Replies are templates filled from
   facts, so the agent cannot announce a change that did not happen.
2. **Identity only from the token.** `company_id` and `employee_id` come from the verified JWT. The chat
   body accepts only `message` (a `company_id` field gets a 422). Model output passes a whitelist, so
   keys like `company_id` or `approved: true` are dropped. Every SQL query filters by the token's
   company. A manager asking for another company's request gets 404, not 403.
3. **Rules in code (`app/rules.py`), two kinds of violation.** `trip_started` and `in_past` are hard.
   `cutoff` and `max_shift` are soft: the agent explains, offers the nearest slot that breaks no rule,
   and offers to ask the manager. A broken rule never leads to a write.
4. **Approval survives restarts.** Escalation writes a `pending_approval` row, then the graph calls
   LangGraph `interrupt()`. The checkpoint lives in Postgres (`PostgresSaver`). The manager endpoint
   finds the thread from the row and resumes it, re-checking the seat and time first because the cab
   may have filled during the wait. A conditional UPDATE stops two managers deciding the same request.
5. **Seats and double submits (bonus).** `UPDATE cabs SET booked = booked + 1 WHERE … AND booked <
   capacity` in one transaction with the roster row locked, plus a `CHECK (booked <= capacity)`
   constraint. A partial unique index allows one pending approval per trip. Each employee's messages
   are serialised with an advisory lock; an `Idempotency-Key` header replays the first response.
6. **Failures become plain sentences.** 2 s DB statement timeout, 4 s LLM timeout. Tool errors, LLM
   errors and a cab filling mid-conversation each get a message saying what happened, that nothing was
   changed, and what to do next. Stack traces are logged server-side, never shown.

## How it stays under 4 LLM calls and 5 s p95

- Only the `understand` step calls the LLM: **1 call** normally, **2** if the first output fails
  validation or the API call errors (one repair/retry). Replies are templates: 0 calls.
- `LLMBudget` enforces a hard cap of 4 per message in code, and each response reports `meta.llm_calls`.
- Messages sent while a request waits for the manager use **0 calls** (status comes from the database).
- One small-model call (Groq `openai/gpt-oss-20b`, `reasoning_effort=low`, JSON mode) plus a few
  indexed queries. Measured with the real model: **p50 0.73 s, p95 1.15 s** (see `EVALUATION.md`).

## Trade-offs

- **Template replies vs LLM-written replies.** Less warm, but always true, no extra LLM call, and
  half the latency. For a transport desk, correct beats chatty.
- **A fixed graph instead of a free tool-calling agent.** A ReAct-style loop is more flexible, but its
  number of LLM calls and tool order are unpredictable, so the 4-call limit could only be hoped for.
  The fixed graph makes tool order testable (`tools-01..03` in the eval).
- **Postgres for everything (data, checkpoints, locks, idempotency).** One dependency instead of
  Postgres plus Redis. Redis would be faster for locks, but this load doesn't need it, and the
  transactional guarantees matter more.
- **The cut-off applies to the earlier of the current and the new time.** An assumption: it stops both
  changing a trip that is about to leave and jumping onto a cab that is about to leave.
