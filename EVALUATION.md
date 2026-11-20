# Evaluation report

## Method

- **Unit and integration tests** (`pytest`, 44 tests) run against a real Postgres with a scripted
  fake LLM (`ScriptedLLM`), so they are deterministic and never fail on an LLM quota error.
- **Evaluation set** (`eval/cases.json`, 25 cases) is replayed by `eval/run_eval.py` on a freshly
  seeded database per case, with the clock frozen at Thu 19 Nov 2026 (14:00 unless the case says
  otherwise). Each case is a multi-step script: employee messages, a service restart, a manager
  decision, an injected tool fault, a cab filling up, or two requests at the same moment.
- **How a case is judged:** by observable outcomes, never by matching LLM wording: the `status` of
  each reply, the employee's roster in the database, request rows by status, the order of tool calls,
  the manager endpoint's HTTP code and outcome, and required or forbidden phrases in the reply (for
  example the desk phone number, or no `GLX` or `Traceback`). A case passes only if every expectation
  holds.
- **Two modes.** `--llm fake` feeds each step's recorded `fake_llm` output and tests everything
  after the model. `--llm real` sends the same messages to the real model and also tests the
  model's understanding, and measures latency including the LLM call. Two cases that need the LLM to
  fail on purpose (`fake_only`) are skipped in real mode.

Output: `eval/eval.jsonl` (one line per case: `id, input, expected, actual, pass, notes`) and
`eval/summary.json`.

## Scenarios covered

| Category | Cases | What is checked |
|---|---|---|
| Happy path | apply-01 | Clear request inside the rules is applied; roster and seat counts update |
| Cut-off rejection | cutoff-01..03 | Same request at the same time: Globex (4 h) rejects, Acme (2 h) applies; Acme rejects at 1.5 h |
| Next option | rule-alt-01, full-01 | Rule break offers nearest compliant slot; full cab offers nearest free slot; employee accepts |
| Clarification | clarify-01..02 | "Can I go a bit late?" gets a question, not a guess; "Make it 7 instead" resolves using the earlier turns |
| Cross-company isolation | isolation-01..03 | Prompt injection to act for a Globex employee; asking about another company's employee; Acme manager deciding a Globex request (404) |
| Resume after restart | resume-01..04 | Approve or reject after a restart; status message while waiting (0 LLM calls); cab fills before approval |
| Tool selection | tools-01..03 | Exact tool order for a change, a status question and a rule break (no write tool) |
| Tool failure | failure-01..03 | Roster timeout, cab-options error, offered cab fills before "yes" |
| LLM failure | failure-04..05 | Quota error; invalid output twice (stops at 2 calls) |
| Concurrency (bonus) | race-01..02 | Two employees, one last seat: exactly one gets it. One employee double-submits: exactly one request |

## Results

**Fake LLM:** 25 / 25 cases pass; pytest 44 / 44 pass. The concurrency and approval tests were
repeated 5 times with no flaky failures.

| Category | Result |
|---|---|
| happy_path | 1/1 |
| cutoff_rejection | 3/3 |
| next_option | 2/2 |
| clarification | 2/2 |
| cross_company_isolation | 3/3 |
| resume_after_restart | 4/4 |
| tool_selection | 3/3 |
| tool_failure | 3/3 |
| llm_failure | 2/2 |
| concurrency_bonus | 2/2 |

Latency in fake mode (no network): p50 24 ms, p95 50 ms per message. This is the cost of our own
code and the database.

**Real LLM (Groq `openai/gpt-oss-20b`, `reasoning_effort=low`):** run with
`EVAL_PACE_S=7 python -m eval.run_eval --llm real`. The pause keeps the free tier's 8,000 tokens/minute
cap from turning into failed calls; it is applied before the timer starts, so latency is unaffected.
Output: `eval/eval_real_llm.jsonl`, `eval/summary_real_llm.json`.

| Metric | Value |
|---|---|
| Cases passed | 22 / 23 (the 2 `fake_only` cases are skipped) |
| p50 latency | 732 ms |
| p95 latency (target < 5000 ms) | 1145 ms |
| Max LLM calls in any message (target ≤ 4) | 1 (last message of each case) |

The one failure, `rule-alt-01`, came from Groq rejecting the first message's API call on both attempts.
It was not a misread: re-running the case alone passed 3 / 3. LLM errors are now logged with the
API's status code.

What the real-model runs found, and what was fixed:

- *"can I leave at 7:30 tonight?"* was read as a **pickup** at 07:30. The prompt now defines leave/go
  home = drop and tonight = PM, and gives an example output.
- *"ok 9:30 works"* right after 21:30 was offered was read as 09:30. Code now maps a time that matches an
  offered slot on a 12-hour clock to that slot (test `test_offered_slot_named_with_wrong_am_pm_is_taken_as_the_offer`).
- *"Change Arjun Rao's (G201) drop to 9 PM"* changed the **asker's own** drop. Another employee's
  data was never touched, but an unrequested change was booked. The prompt now classifies any request
  about another person as `other`, and the reply says "I can only change your own trips."
- One transient API error used to end the message. It now uses the second attempt as a retry (still at
  most 2 calls).

**Held-out phrasings** (`eval/heldout.py`, against the running app, results in
`eval/heldout_results.jsonl`): 12 messages that are not in `cases.json` (slang such as "pls shift tmrw
morning pickup to 10", "head home at half past seven", "8 o'clock", a fake `SYSTEM:` prompt injection,
another employee's id, cancelling after an offer). **12 / 12 pass**, judged by reply status and the
roster row in the database; latency 0.8 to 1.6 s.

## Known failures and limitations

- **Real-model understanding is the weakest link.** In fake mode the model's output is assumed
  correct, so those results test everything *after* the model. Expected weak spots with a small
  model: AM/PM for bare numbers ("make it 8"), and telling "yes" apart from "yes, send to manager"
  when both options were offered. If the choice is unclear the agent asks again rather than guessing.
- Pending requests do not expire on their own; a late approval is reported as "too late".
- If the process is killed exactly in the middle of a resume, the request can stay claimed.
- Only today's and tomorrow's trips; the conversation history is not trimmed.

## A trade-off I made

**Template replies instead of LLM-written replies.** A second LLM call could phrase replies more
naturally. I chose templates built from the facts that code decided: the reply can never claim a
change that didn't happen, each message needs 1 LLM call instead of 2 (easily under the cap of 4),
and latency roughly halves. The cost is stiffer wording. For a tired employee at 8 PM, a short,
correct sentence with the exact time and next step is worth more than a warm one that might be
wrong.
