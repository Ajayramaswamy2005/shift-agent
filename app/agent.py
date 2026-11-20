"""The shift-change agent as a LangGraph graph.

    START -> understand -> resolve -> evaluate -> apply -----------------> finish -> END
                               |          |                                  ^
                               |          +-- (offer / reject / clarify) ---+
                               +-- escalate -> wait_for_manager -> after_decision
                                                  (interrupt)

Only `understand` calls the LLM (max 2 calls: one try + one repair). Everything that
decides an outcome (which trip, rules, seats, apply, escalate) is plain Python, so the
model cannot approve, book or leak anything. Replies are templates built from facts,
so the agent can never "confirm" a change that did not happen.

State is checkpointed in Postgres, so a paused approval survives a restart.
"""
import logging
import operator
from datetime import date, datetime, time, timedelta
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app import clock
from app.auth import Identity
from app.clock import IST
from app.llm import LLM, LLMError, build_user_prompt, understand as llm_understand
from app.rules import Policy, check_change
from app.tools import CabFull, ToolError, TransportTools, TripChanged, trip_datetime

log = logging.getLogger("shift-agent")
MAX_LLM_CALLS_PER_MESSAGE = 4
UNDERSTAND_ATTEMPTS = 2  # 1 try + 1 repair; we never get near the cap of 4


class LLMBudget:
    """Hard cap on LLM calls for one employee message. Enforced in code, not hoped for."""

    def __init__(self, llm: LLM, max_calls: int = MAX_LLM_CALLS_PER_MESSAGE):
        self.llm, self.max_calls, self.calls = llm, max_calls, 0

    def complete_json(self, system: str, user: str) -> str:
        if self.calls >= self.max_calls:
            raise LLMError("LLM call budget for this message exhausted")
        self.calls += 1
        return self.llm.complete_json(system, user)


class AgentState(TypedDict, total=False):
    company_id: str            # from the JWT, set by the API on every call
    employee_id: str           # from the JWT, set by the API on every call
    messages: Annotated[list[dict], operator.add]
    user_text: str
    trips: list[dict]
    parsed: dict | None
    draft: dict                # what we know so far across turns: direction, date, new_time
    offer: dict | None         # the option we last offered (alternative slot / manager approval)
    target: dict | None
    pending: dict | None       # request waiting for the manager
    decision: dict | None
    result: dict               # {"status", "reply", "request_id"?}
    route: str | None


# ---- wording helpers ---------------------------------------------------------------
def _day(d: str, today: date) -> str:
    dd = date.fromisoformat(d)
    if dd == today:
        return "today"
    if dd == today + timedelta(days=1):
        return f"tomorrow ({dd:%a %d %b})"
    return f"on {dd:%a %d %b}"


def _same_clock_time(a: str, b: str) -> bool:
    """True if two HH:MM times are the same on a 12-hour clock (09:30 vs 21:30)."""
    ha, ma = map(int, a.split(":"))
    hb, mb = map(int, b.split(":"))
    return ma == mb and ha % 12 == hb % 12


def _describe(trip: dict, today: date) -> str:
    return f"your {trip['direction']} {_day(trip['date'], today)} at {trip['slot']}"


def _list_trips(trips: list[dict], today: date) -> str:
    return "; ".join(f"{t['direction']} {_day(t['date'], today)} at {t['slot']}" for t in trips)


def _done(status: str, reply: str, clear: bool = True, **extra) -> dict:
    """A terminal result. clear=True forgets the half-finished request and any open offer."""
    out: dict[str, Any] = {"result": {"status": status, "reply": reply, **extra}, "route": "finish"}
    if clear:
        out.update(draft={}, offer=None)
    return out


class ShiftAgent:
    def __init__(self, llm: LLM, tools: TransportTools, checkpointer):
        self.llm, self.tools = llm, tools
        self._desk: dict[str, str] = {}
        self.graph = self._build().compile(checkpointer=checkpointer)

    # ---- small utilities -----------------------------------------------------------
    @staticmethod
    def _ident(state: AgentState) -> Identity:
        return Identity(state["company_id"], state["employee_id"], "employee")

    def _policy(self, ident: Identity) -> Policy:
        p = self.tools.get_policy(ident)
        self._desk[ident.company_id] = p.desk_phone
        return p

    def _desk_line(self, company_id: str) -> str:
        phone = self._desk.get(company_id)
        if phone is None:
            try:   # not cached yet (e.g. the very first message failed before the policy was read)
                phone = self.tools.get_policy(Identity(company_id, "", "employee")).desk_phone
                self._desk[company_id] = phone
            except Exception:
                phone = None
        return f"the transport desk at {phone}" if phone else "the transport desk"

    def _tool_down(self, company_id: str) -> dict:
        return _done("error",
                     "I couldn't reach the transport system just now, so nothing was changed. "
                     f"Please try again in a minute, or call {self._desk_line(company_id)}.",
                     clear=False)

    def _nearest_ok(self, options: list[dict], trip: dict, wanted: datetime, policy: Policy,
                    now: datetime, exclude: set[str]) -> dict | None:
        """Nearest cab to the wanted time that has a seat and breaks no rule."""
        current = trip_datetime(trip)
        best = None
        for o in options:
            if o["cab_id"] in exclude or o["seats_free"] <= 0:
                continue
            dt = datetime.combine(date.fromisoformat(trip["date"]), time.fromisoformat(o["slot"]), IST)
            if check_change(policy, now, current, dt):
                continue
            if best is None or abs(dt - wanted) < best[0]:
                best = (abs(dt - wanted), o)
        return best[1] if best else None

    # ---- nodes -------------------------------------------------------------------
    def understand(self, state: AgentState, config: RunnableConfig) -> dict:
        ident, now = self._ident(state), clock.now()
        try:
            trips = self.tools.get_upcoming_trips(ident, now)
        except ToolError:
            return self._tool_down(ident.company_id)
        prompt = build_user_prompt(now.date(), f"{now:%H:%M}", trips, state.get("draft") or {},
                                   state.get("offer"), state.get("messages", [])[:-1], state["user_text"])
        budget: LLMBudget = config["configurable"]["llm_budget"]
        try:
            parsed = llm_understand(budget, prompt, max_attempts=UNDERSTAND_ATTEMPTS)
        except LLMError as e:
            log.warning("understand failed: %s", e)
            return _done("error",
                         "Sorry, I'm having trouble understanding messages right now, so nothing was changed. "
                         f"Please try again in a minute, or call {self._desk_line(ident.company_id)}.",
                         clear=False)
        return {"trips": trips, "parsed": parsed, "route": "resolve"}

    def resolve(self, state: AgentState) -> dict:
        """Work out which trip and which time the employee means. Ask if unclear; never guess."""
        ident, now = self._ident(state), clock.now()
        today = now.date()
        p, trips = state["parsed"], state["trips"]
        draft, offer = dict(state.get("draft") or {}), state.get("offer")
        intent = p["intent"]

        if intent == "other":
            return _done("info", "I can only change your own trips. I can help you change the time of your "
                                 "pickup or drop for today or tomorrow, for example: \"Can I take the 8 PM drop "
                                 "tonight?\"", clear=False)

        if intent == "status":
            try:
                pending = self.tools.get_pending_request(ident)
            except ToolError:
                return self._tool_down(ident.company_id)
            text = (f"Your upcoming trips: {_list_trips(trips, today)}." if trips
                    else "You have no upcoming trips today or tomorrow.")
            if pending:
                text += f" One change, to {pending['to_slot']:%H:%M}, is waiting for your manager's approval."
            return _done("info", text, clear=False)

        if intent == "decline":
            return _done("info", "OK, I've left your trips as they are.")

        if intent == "accept_offer":
            if not offer:
                return _done("clarify", "I don't have an option open for you right now. "
                                        "Which trip do you want to change, and to what time?", clear=False)
            options = [k for k in ("alternative", "manager") if offer.get(k)]
            choice = p["choice"] if p["choice"] in options else (options[0] if len(options) == 1 else None)
            if choice is None:
                return _done("clarify", f"Do you want the {offer['alternative']['slot']} slot, or should I ask "
                                        f"your manager to approve {offer['manager']['slot']}?", clear=False)
            trip = offer["trip"]
            if choice == "alternative":
                return {"target": {"trip": trip, "new_time": offer["alternative"]["slot"]}, "route": "evaluate"}
            return {"target": {"trip": trip, "cab": offer["manager"], "violations": offer["violations"]},
                    "route": "escalate"}

        # AM/PM guard: "9:30 works" right after we offered 21:30 means the offered slot, not 09:30.
        if offer and p["new_time"]:
            for key in ("alternative", "manager"):
                slot = (offer.get(key) or {}).get("slot")
                if slot and slot != p["new_time"] and _same_clock_time(slot, p["new_time"]):
                    p = dict(p, new_time=slot)
            alt = (offer.get("alternative") or {}).get("slot")
            if alt and p["new_time"] == alt:
                return {"target": {"trip": offer["trip"], "new_time": alt}, "route": "evaluate"}

        # intent == "change_time": merge what we learned now with what we knew before.
        for k in ("direction", "date", "new_time"):
            if p[k]:
                draft[k] = p[k]
        if draft.get("date") and draft["date"] not in (today.isoformat(), (today + timedelta(days=1)).isoformat()):
            return _done("rejected", "I can only change trips for today and tomorrow. "
                                     f"For other dates please contact {self._desk_line(ident.company_id)}.")
        if not trips:
            return _done("info", "You have no upcoming trips today or tomorrow, so there's nothing to change.")
        matches = [t for t in trips if (not draft.get("direction") or t["direction"] == draft["direction"])
                   and (not draft.get("date") or t["date"] == draft["date"])]
        if not matches:
            draft.pop("direction", None), draft.pop("date", None)
            return _done("clarify", f"I couldn't find that trip. Your upcoming trips are: {_list_trips(trips, today)}. "
                                    "Which one do you want to change?", clear=False) | {"draft": draft}
        if len(matches) > 1:
            return _done("clarify", f"Which trip do you mean: {_list_trips(matches, today)}?",
                         clear=False) | {"draft": draft}
        trip = matches[0]
        draft.update(direction=trip["direction"], date=trip["date"])
        if not draft.get("new_time"):
            return _done("clarify", f"Sure. What time would you like instead? Right now {_describe(trip, today)}.",
                         clear=False) | {"draft": draft}
        return {"draft": draft, "target": {"trip": trip, "new_time": draft["new_time"]}, "route": "evaluate"}

    def evaluate(self, state: AgentState) -> dict:
        """Apply this company's rules and look for a seat. Code decides; the LLM is not involved."""
        ident, now = self._ident(state), clock.now()
        today = now.date()
        trip, new_time = state["target"]["trip"], state["target"]["new_time"]
        if new_time == trip["slot"]:
            return _done("no_change", f"Your {trip['direction']} {_day(trip['date'], today)} is already at "
                                     f"{trip['slot']}. Nothing to change.")
        try:
            policy = self._policy(ident)
            new_dt = datetime.combine(date.fromisoformat(trip["date"]), time.fromisoformat(new_time), IST)
            violations = check_change(policy, now, trip_datetime(trip), new_dt)
            hard = [v for v in violations if not v.escalatable]
            if hard:
                return _done("rejected", f"I can't change {_describe(trip, today)}: {hard[0].message}. "
                                         f"If you need help, call {self._desk_line(ident.company_id)}.")
            options = self.tools.find_cab_options(ident, trip)
        except ToolError:
            return self._tool_down(ident.company_id)

        exact = next((o for o in options if o["slot"] == new_time and o["cab_id"] != trip["cab_id"]), None)
        alt = self._nearest_ok(options, trip, new_dt, policy, now,
                               exclude={trip["cab_id"]} | ({exact["cab_id"]} if exact else set()))

        if exact and exact["seats_free"] > 0 and not violations:
            return {"target": {"trip": trip, "cab": exact}, "route": "apply"}

        keep = f"Your {trip['slot']} {trip['direction']} stays booked until you choose."
        if exact and exact["seats_free"] > 0:   # seat exists, but a rule is broken -> offer manager approval
            reasons = "; also ".join(v.message for v in violations)
            text = f"I can't move {_describe(trip, today)} to {new_time} myself: {reasons}."
            if alt:
                text += (f" The nearest time that needs no approval is {alt['slot']}. Reply \"yes\" to take "
                         f"{alt['slot']}, or \"send to manager\" to ask your transport manager to approve {new_time}.")
            else:
                text += (f" Reply \"send to manager\" to ask your transport manager to approve {new_time}, "
                         f"or \"no\" to keep {trip['slot']}.")
            offer = {"trip": trip, "alternative": alt, "manager": exact,
                     "violations": [{"code": v.code, "message": v.message} for v in violations]}
            return _done("rule_violation", f"{text} {keep}", clear=False) | {"offer": offer}

        why = (f"The {new_time} cab is full." if exact
               else f"There's no {trip['direction']} cab at {new_time} on your route.")
        if alt:
            return _done("unavailable", f"{why} The nearest slot with a free seat is {alt['slot']}. "
                                        f"Shall I move you to {alt['slot']}? {keep}",
                         clear=False) | {"offer": {"trip": trip, "alternative": alt}}
        return _done("unavailable", f"{why} I couldn't find another slot with a free seat that you can switch to. "
                                    f"Your {trip['slot']} {trip['direction']} is unchanged. "
                                    f"For other options call {self._desk_line(ident.company_id)}.")

    def apply(self, state: AgentState) -> dict:
        ident, today = self._ident(state), clock.now().date()
        trip, cab = state["target"]["trip"], state["target"]["cab"]
        try:
            r = self.tools.apply_change(ident, trip, cab["cab_id"], ident.thread_id)
        except CabFull:
            # Filled up while the employee was deciding: say so and offer the next option.
            return self._refilled(state, trip, cab)
        except TripChanged:
            return _done("error", "Your trip was changed by another request a moment ago, so I didn't change "
                                  "anything. Ask me again if you still want a different time.")
        except ToolError:
            return self._tool_down(ident.company_id)
        if r["already_done"]:
            return _done("no_change", f"You're already on the {cab['slot']} {trip['direction']}. Nothing to change.")
        return _done("applied", f"Done. Your {trip['direction']} {_day(trip['date'], today)} is now at "
                                f"{cab['slot']} instead of {trip['slot']} (cab {cab['cab_id']}). "
                                f"Reference: {r['request_id'][:8]}.", request_id=r["request_id"])

    def _refilled(self, state: AgentState, trip: dict, cab: dict) -> dict:
        ident, now = self._ident(state), clock.now()
        try:
            policy = self._policy(ident)
            options = self.tools.find_cab_options(ident, trip)
        except ToolError:
            return self._tool_down(ident.company_id)
        wanted = datetime.combine(date.fromisoformat(trip["date"]), time.fromisoformat(cab["slot"]), IST)
        alt = self._nearest_ok(options, trip, wanted, policy, now, exclude={trip["cab_id"], cab["cab_id"]})
        text = f"Sorry, the {cab['slot']} cab filled up while we were talking. Your {trip['slot']} {trip['direction']} is unchanged."
        if alt:
            return _done("unavailable", f"{text} The nearest slot with a free seat now is {alt['slot']}. "
                                        f"Shall I move you to {alt['slot']}?",
                         clear=False) | {"offer": {"trip": trip, "alternative": alt}}
        return _done("unavailable", f"{text} No other slot has a free seat right now. "
                                    f"Please call {self._desk_line(ident.company_id)}.")

    def escalate(self, state: AgentState) -> dict:
        ident, today = self._ident(state), clock.now().date()
        t = state["target"]
        try:
            rid = self.tools.create_approval_request(ident, t["trip"], t["cab"]["cab_id"], t["violations"],
                                                     ident.thread_id)
        except ToolError:
            return self._tool_down(ident.company_id)
        text = (f"I've sent your request to move {_describe(t['trip'], today)} to {t['cab']['slot']} to your "
                f"transport manager. Your {t['trip']['slot']} {t['trip']['direction']} stays booked until they "
                f"decide. I'll post their answer here.")
        return {"pending": {"request_id": rid, "trip": t["trip"], "cab": t["cab"]},
                "result": {"status": "pending_approval", "reply": text, "request_id": rid},
                "messages": [{"role": "assistant", "content": text}],
                "draft": {}, "offer": None, "route": "wait"}

    def wait_for_manager(self, state: AgentState) -> dict:
        # Pauses the graph. The checkpoint is in Postgres, so this survives a restart.
        decision = interrupt({"request_id": state["pending"]["request_id"]})
        return {"decision": decision}

    def after_decision(self, state: AgentState) -> dict:
        ident, now = self._ident(state), clock.now()
        today = now.date()
        d, p = state["decision"], state["pending"]
        trip, cab, rid = p["trip"], p["cab"], p["request_id"]
        desk = self._desk_line(ident.company_id)
        new_dt = datetime.combine(date.fromisoformat(trip["date"]), time.fromisoformat(cab["slot"]), IST)
        if not d.get("approved"):
            note = f" Their note: \"{d['note']}\"." if d.get("note") else ""
            self.tools.close_request(ident.company_id, rid, "rejected", d["manager_id"], d.get("note") or "")
            status, text = "declined", (f"Your transport manager didn't approve moving {_describe(trip, today)} "
                                        f"to {cab['slot']}.{note} Your {trip['slot']} {trip['direction']} is unchanged.")
        elif min(trip_datetime(trip), new_dt) <= now:
            self.tools.close_request(ident.company_id, rid, "failed", d["manager_id"], "approved too late")
            status, text = "failed", (f"Your manager approved the change to {cab['slot']}, but it came too late: "
                                      f"the trip time has passed. Please call {desk}.")
        else:
            try:
                self.tools.apply_change(ident, trip, cab["cab_id"], ident.thread_id,
                                        request_id=rid, decided_by=d["manager_id"])
                status, text = "approved", (f"Good news: your transport manager approved it. Your "
                                            f"{trip['direction']} {_day(trip['date'], today)} is now at "
                                            f"{cab['slot']} (cab {cab['cab_id']}).")
            except (CabFull, TripChanged):
                self.tools.close_request(ident.company_id, rid, "failed", d["manager_id"], "cab full at approval")
                status, text = "failed", (f"Your manager approved moving to {cab['slot']}, but that cab filled up "
                                          f"while the request was waiting. Your {trip['slot']} {trip['direction']} "
                                          f"is unchanged. Tell me another time and I'll check, or call {desk}.")
        self.tools.notify(ident.company_id, ident.user_id, text)
        return {"result": {"status": status, "reply": text, "request_id": rid},
                "pending": None, "decision": None, "route": "finish"}

    def finish(self, state: AgentState) -> dict:
        return {"messages": [{"role": "assistant", "content": state["result"]["reply"]}]}

    # ---- graph ---------------------------------------------------------------------
    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        for name in ("understand", "resolve", "evaluate", "apply", "escalate", "wait_for_manager",
                     "after_decision", "finish"):
            g.add_node(name, getattr(self, name))
        go = lambda s: s.get("route") or "finish"  # noqa: E731
        g.add_edge(START, "understand")
        g.add_conditional_edges("understand", go, ["resolve", "finish"])
        g.add_conditional_edges("resolve", go, ["evaluate", "escalate", "finish"])
        g.add_conditional_edges("evaluate", go, ["apply", "finish"])
        g.add_edge("apply", "finish")
        g.add_conditional_edges("escalate", lambda s: "wait_for_manager" if s.get("route") == "wait" else "finish",
                                ["wait_for_manager", "finish"])
        g.add_edge("wait_for_manager", "after_decision")
        g.add_edge("after_decision", "finish")
        g.add_edge("finish", END)
        return g

    # ---- entry points used by the API ---------------------------------------------------
    @staticmethod
    def _config(thread_id: str, budget: LLMBudget | None = None) -> dict:
        return {"configurable": {"thread_id": thread_id, "llm_budget": budget}}

    def is_waiting(self, thread_id: str) -> bool:
        return bool(self.graph.get_state(self._config(thread_id)).next)

    def handle_message(self, ident: Identity, text: str) -> dict:
        today = clock.now().date()
        snap = self.graph.get_state(self._config(ident.thread_id))
        if snap.next:  # paused for the manager: answer from facts, don't start a new run
            p = snap.values["pending"]
            age = clock.now() - _created_at(self.tools, ident)
            mins = max(0, int(age.total_seconds() // 60))
            waited = f"{mins // 60} h {mins % 60} min" if mins >= 60 else f"{mins} min"
            reply = (f"Your request to move {_describe(p['trip'], today)} to {p['cab']['slot']} is still waiting "
                     f"for your transport manager (sent {waited} ago). Your {p['trip']['slot']} "
                     f"{p['trip']['direction']} stays booked until they decide, and I'll post the answer here.")
            if mins >= 60:
                reply += f" If it's urgent, call {self._desk_line(ident.company_id)}."
            return {"status": "pending_approval", "reply": reply, "request_id": p["request_id"],
                    "meta": {"llm_calls": 0}}

        budget = LLMBudget(self.llm)
        out = self.graph.invoke(
            {"company_id": ident.company_id, "employee_id": ident.user_id, "user_text": text,
             "messages": [{"role": "user", "content": text}],
             "trips": [], "parsed": None, "target": None, "result": {}, "route": None},
            self._config(ident.thread_id, budget),
        )
        r = out["result"]
        return {"status": r["status"], "reply": r["reply"], "request_id": r.get("request_id"),
                "meta": {"llm_calls": budget.calls}}

    def resume(self, thread_id: str, decision: dict) -> dict:
        out = self.graph.invoke(Command(resume=decision), self._config(thread_id, LLMBudget(self.llm, 0)))
        return out["result"]

    def history(self, thread_id: str) -> list[dict]:
        return self.graph.get_state(self._config(thread_id)).values.get("messages", [])


def _created_at(tools: TransportTools, ident: Identity) -> datetime:
    try:
        req = tools.get_pending_request(ident)
    except ToolError:
        req = None
    return req["created_at"].astimezone(IST) if req else clock.now()
