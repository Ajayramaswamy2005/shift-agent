"""HTTP API.

  POST /dev/token                              mock JWT issuer (looks the user up in the DB)
  POST /chat                                   employee chat (JWT: employee)
  GET  /chat/history                           employee's conversation
  GET  /manager/requests                       pending approvals for the manager's company
  POST /manager/requests/{id}/decision         approve / reject -> resumes the paused conversation
  GET  /health
"""
import logging
import re
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from app import auth, clock, config
from app.agent import ShiftAgent
from app.auth import Identity
from app.db import make_pool
from app.llm import LLM, OpenAICompatibleLLM
from app.seed import seed_if_empty
from app.tools import TransportTools

log = logging.getLogger("shift-agent")
GENERIC_ERROR = ("Something went wrong on our side and nothing was changed. "
                 "Please try again in a minute, or call the transport desk.")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class Busy(Exception):
    pass


class ChatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")   # a "company_id" in the body is rejected, not used
    message: str = Field(min_length=1, max_length=1000)


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool
    note: str | None = Field(default=None, max_length=500)


class TokenIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=50)


@contextmanager
def conversation_lock(pool, thread_id: str):
    """One message at a time per employee, across processes (Postgres advisory lock).
    Waits at most the DB statement timeout, then reports 'busy' instead of hanging."""
    with pool.connection() as conn:
        try:
            conn.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (thread_id,))
        except psycopg.errors.QueryCanceled:
            raise Busy()
        try:
            yield
        finally:
            conn.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (thread_id,))


def create_app(llm: LLM | None = None, database_url: str | None = None, seed: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = make_pool(database_url)
        if seed:
            seed_if_empty(pool, clock.now().date())
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        tools = TransportTools(pool)
        app.state.pool, app.state.tools = pool, tools
        app.state.agent = ShiftAgent(llm or OpenAICompatibleLLM.from_env(), tools, checkpointer)
        yield
        pool.close()

    app = FastAPI(title="Shift-Change Request Agent", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def bad_request(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={
            "status": "error",
            "reply": "I couldn't read that request. Send JSON like {\"message\": \"...\"} (up to 1000 characters).",
        })

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")  # full trace goes to the server log, never to the user
        return JSONResponse(status_code=500, content={"status": "error", "reply": GENERIC_ERROR})

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/")
    def ui():
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    @app.post("/dev/token")
    def dev_token(body: TokenIn, request: Request):
        if not config.dev_token_issuer_enabled():
            raise HTTPException(404, "Not found")
        with request.app.state.pool.connection() as conn:
            u = conn.execute("SELECT id, company_id, role, name FROM employees WHERE id = %s",
                             (body.user_id,)).fetchone()
        if not u:
            raise HTTPException(404, "Unknown user")
        return {"token": auth.issue_token(u["company_id"], u["id"], u["role"]),
                "company_id": u["company_id"], "role": u["role"], "name": u["name"]}

    @app.post("/chat")
    def chat(body: ChatIn, request: Request, ident: Identity = Depends(auth.employee),
             idempotency_key: str | None = Header(default=None, max_length=100)):
        st = request.app.state
        try:
            with conversation_lock(st.pool, ident.thread_id):
                if idempotency_key:
                    with st.pool.connection() as conn:
                        hit = conn.execute(
                            "SELECT response FROM chat_idempotency WHERE company_id=%s AND employee_id=%s AND key=%s",
                            (ident.company_id, ident.user_id, idempotency_key)).fetchone()
                    if hit:
                        return hit["response"] | {"replayed": True}
                updates = st.tools.pop_notifications(ident)
                resp = st.agent.handle_message(ident, body.message) | {"updates": updates}
                if idempotency_key:
                    with st.pool.connection() as conn:
                        conn.execute("INSERT INTO chat_idempotency (company_id, employee_id, key, response) "
                                     "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                                     (ident.company_id, ident.user_id, idempotency_key, Jsonb(resp)))
                return resp
        except Busy:
            return {"status": "busy", "reply": "I'm still working on your previous message. "
                                               "Please wait a few seconds and send it again.", "updates": []}
        except Exception:
            log.exception("chat failed")
            return JSONResponse(status_code=500, content={"status": "error", "reply": GENERIC_ERROR, "updates": []})

    @app.get("/chat/history")
    def history(request: Request, ident: Identity = Depends(auth.employee)):
        return {"messages": request.app.state.agent.history(ident.thread_id)}

    @app.get("/manager/requests")
    def pending(request: Request, mgr: Identity = Depends(auth.manager)):
        rows = request.app.state.tools.list_pending(mgr.company_id)
        return {"requests": [dict(r, id=str(r["id"])) for r in rows]}

    @app.post("/manager/requests/{request_id}/decision")
    def decide(request_id: str, body: DecisionIn, request: Request, mgr: Identity = Depends(auth.manager)):
        st = request.app.state
        if not UUID_RE.match(request_id):
            raise HTTPException(404, "No such request.")
        req = st.tools.claim_request(mgr.company_id, request_id, mgr.user_id)   # scoped to manager's company
        if req is None:
            if st.tools.get_request(mgr.company_id, request_id) is None:
                raise HTTPException(404, "No such request.")   # includes other companies' requests
            raise HTTPException(409, "This request has already been decided.")
        if not st.agent.is_waiting(req["thread_id"]):
            st.tools.unclaim_request(mgr.company_id, request_id)
            raise HTTPException(409, "The conversation for this request is not waiting for a decision.")
        try:
            with conversation_lock(st.pool, req["thread_id"]):
                result = st.agent.resume(req["thread_id"], {"approved": body.approved, "note": body.note,
                                                            "manager_id": mgr.user_id})
        except Exception:
            log.exception("resume failed")
            st.tools.unclaim_request(mgr.company_id, request_id)
            raise HTTPException(503, "Couldn't record the decision right now. Please try again.")
        return {"request_id": request_id, "outcome": result["status"], "employee_message": result["reply"]}

    return app


app = create_app()
