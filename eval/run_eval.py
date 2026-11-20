"""Run the evaluation set and write eval/eval.jsonl (id, input, expected, actual, pass, notes).

  python -m eval.run_eval              # fake LLM (deterministic, uses each step's fake_llm outputs)
  python -m eval.run_eval --llm real   # real LLM from LLM_* env vars; also reports p95 latency

Each case runs on a freshly seeded database (a separate *_eval database, never the demo one),
with the clock frozen at Thursday 2026-11-19 (14:00 unless the case sets "now").
A case passes only if every expectation holds. Judging is by observable state (status,
roster in the DB, request rows, tool calls), not by matching LLM wording.
"""
import argparse
import json
import os
import statistics
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import psycopg
from fastapi.testclient import TestClient

from app import clock
from app.db import make_pool
from app.llm import LLMError, OpenAICompatibleLLM, ScriptedLLM
from app.main import create_app
from app.seed import reset

HERE = Path(__file__).parent
DAY = datetime(2026, 11, 19, tzinfo=clock.IST)


def eval_db_url() -> str:
    base = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/shift")
    url = os.getenv("EVAL_DATABASE_URL", base.rsplit("/", 1)[0] + "/shift_eval")
    name, admin = url.rsplit("/", 1)[1], url.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            conn.execute(f'CREATE DATABASE "{name}"')
    return url


class Runner:
    def __init__(self, url: str, real: bool):
        self.url, self.real = url, real
        self.latencies: list[float] = []

    def _start(self):
        self.client = TestClient(create_app(llm=self.llm, database_url=self.url, seed=False))
        self.client.__enter__()
        self.tokens = {}

    def _stop(self):
        self.client.__exit__(None, None, None)

    def _auth(self, user):
        if user not in self.tokens:
            self.tokens[user] = self.client.post("/dev/token", json={"user_id": user}).json()["token"]
        return {"Authorization": f"Bearer {self.tokens[user]}"}

    def _push_fake(self, outputs):
        if self.real:
            return
        for o in outputs:
            self.llm.push(LLMError("simulated quota error") if o == "__error__" else o)

    def _chat(self, user, text):
        if self.real:   # free-tier LLM keys have a tokens-per-minute cap; pace calls so it isn't hit
            _time.sleep(float(os.getenv("EVAL_PACE_S", "0")))
        t0 = _time.perf_counter()
        r = self.client.post("/chat", json={"message": text}, headers=self._auth(user)).json()
        self.latencies.append((_time.perf_counter() - t0) * 1000)
        return r

    def _q(self, sql, *params):
        with self.client.app.state.pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    def run_case(self, case: dict) -> dict:
        hh, mm = map(int, case.get("now", "14:00").split(":"))
        clock.freeze(DAY.replace(hour=hh, minute=mm))
        pool = make_pool(self.url)
        reset(pool, DAY.date())
        pool.close()
        self.llm = OpenAICompatibleLLM.from_env() if self.real else ScriptedLLM()
        self._start()
        statuses, transcript = [], []
        last, actual = None, {}
        try:
            for step in case["steps"]:
                if "say" in step:
                    self._push_fake(step.get("fake_llm", []))
                    self.client.app.state.tools.calls.clear()
                    last = self._chat(step["user"], step["say"])
                    statuses.append(last.get("status"))
                    actual["tools"] = list(self.client.app.state.tools.calls)
                    transcript.append(f"{step['user']}: {step['say']}")
                elif "race" in step:
                    self._push_fake(step.get("fake_llm", []))
                    for s in step["race"]:
                        self._auth(s["user"])
                    with ThreadPoolExecutor(len(step["race"])) as ex:
                        res = list(ex.map(lambda s: self._chat(s["user"], s["say"]), step["race"]))
                    actual["race_statuses"] = sorted(r.get("status") for r in res)
                    transcript.append("[at the same moment] " + " || ".join(f"{s['user']}: {s['say']}" for s in step["race"]))
                elif step.get("restart"):
                    self._stop()
                    self._start()
                    transcript.append("[service restart]")
                elif "manager" in step:
                    pending = self._q("SELECT id FROM shift_requests WHERE status = 'pending_approval' "
                                      "ORDER BY created_at DESC LIMIT 1")
                    rid = str(pending[0]["id"]) if pending else "00000000-0000-0000-0000-000000000000"
                    r = self.client.post(f"/manager/requests/{rid}/decision", headers=self._auth(step["manager"]),
                                         json={"approved": step["decision"] == "approve", "note": step.get("note")})
                    actual["manager_http"] = r.status_code
                    actual["manager_outcome"] = r.json().get("outcome")
                    transcript.append(f"[{step['manager']} {step['decision']}s]")
                elif "fault" in step:
                    self.client.app.state.tools.faults.update(step["fault"])
                    transcript.append(f"[fault injected: {step['fault']}]")
                elif "fill_cab" in step:
                    self._q("UPDATE cabs SET booked = capacity WHERE id = %s RETURNING id", step["fill_cab"])
                    transcript.append(f"[cab {step['fill_cab']} fills up]")

            exp = case["expect"]
            if last:
                actual["last_reply"] = last.get("reply")
                actual["last_llm_calls"] = (last.get("meta") or {}).get("llm_calls")
                actual["updates"] = last.get("updates", [])
            actual["statuses"] = statuses
            if "slots" in exp:
                actual["slots"] = {}
                for emp in exp["slots"]:
                    row = self._q("SELECT c.slot FROM roster r JOIN cabs c ON c.id = r.cab_id WHERE r.employee_id = %s "
                                  "AND r.trip_date = %s AND r.direction = 'drop'", emp, DAY.date())
                    actual["slots"][emp] = row[0]["slot"].strftime("%H:%M")
            if "requests" in exp:
                actual["requests"] = {s: self._q("SELECT count(*) AS n FROM shift_requests WHERE status = %s", s)[0]["n"]
                                      for s in exp["requests"]}
            if "no_requests_for_company" in exp:
                actual["requests_for_company"] = self._q(
                    "SELECT count(*) AS n FROM shift_requests WHERE company_id = %s", exp["no_requests_for_company"])[0]["n"]
            failures = judge(exp, actual)
        except Exception as e:  # a crash in the runner is a failed case, not a crashed run
            failures = [f"runner exception: {type(e).__name__}: {e}"]
        finally:
            self._stop()
            clock.freeze(None)

        return {
            "id": case["id"],
            "input": " | ".join(transcript),
            "expected": case["expect"],
            "actual": {k: v for k, v in actual.items() if k != "updates" or v},
            "pass": not failures,
            "notes": f"[{case['category']}] {case['notes']}" + (f" FAILED: {'; '.join(failures)}" if failures else ""),
        }


def judge(exp: dict, act: dict) -> list[str]:
    f = []
    reply = (act.get("last_reply") or "")
    if "statuses" in exp and act.get("statuses") != exp["statuses"]:
        f.append(f"statuses {act.get('statuses')} != {exp['statuses']}")
    for s in exp.get("reply_contains", []):
        if s.lower() not in reply.lower():
            f.append(f"reply missing '{s}'")
    for s in exp.get("reply_not_contains", []):
        if s.lower() in reply.lower():
            f.append(f"reply must not contain '{s}'")
    for k in ("slots", "requests", "tools", "race_statuses", "manager_http", "manager_outcome", "last_llm_calls"):
        if k in exp and act.get(k) != exp[k]:
            f.append(f"{k} {act.get(k)} != {exp[k]}")
    if "no_requests_for_company" in exp and act.get("requests_for_company"):
        f.append(f"{act['requests_for_company']} request(s) created for {exp['no_requests_for_company']}")
    if "updates_contain" in exp and not any(exp["updates_contain"] in u for u in act.get("updates", [])):
        f.append(f"no update containing '{exp['updates_contain']}'")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["fake", "real"], default="fake")
    ap.add_argument("--only", help="comma-separated case ids to run (for debugging)")
    args = ap.parse_args()
    real = args.llm == "real"
    cases = json.loads((HERE / "cases.json").read_text())
    if real:
        cases = [c for c in cases if not c.get("fake_only")]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only.split(",")]
    runner = Runner(eval_db_url(), real)
    results = [runner.run_case(c) for c in cases]
    if args.only:   # debugging run: print, don't overwrite the full results
        for r in results:
            print(json.dumps(r, ensure_ascii=False))
        raise SystemExit(0 if all(r["pass"] for r in results) else 1)

    out = HERE / ("eval_real_llm.jsonl" if real else "eval.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))

    by_cat: dict[str, list[bool]] = {}
    for c, r in zip(cases, results):
        by_cat.setdefault(c["category"], []).append(r["pass"])
    lat = sorted(runner.latencies)
    p95 = lat[max(0, int(round(0.95 * len(lat))) - 1)] if lat else None
    summary = {
        "llm": args.llm,
        "passed": sum(r["pass"] for r in results), "total": len(results),
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in by_cat.items()},
        "latency_ms": {"messages": len(lat), "p50": round(statistics.median(lat), 1) if lat else None,
                       "p95": round(p95, 1) if p95 else None, "max": round(lat[-1], 1) if lat else None},
        "failed": [r["id"] for r in results if not r["pass"]],
    }
    (HERE / ("summary_real_llm.json" if real else "summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    raise SystemExit(0 if summary["passed"] == summary["total"] else 1)


if __name__ == "__main__":
    main()
