from pathlib import Path

from psycopg import ClientCursor
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app import config

SCHEMA = Path(__file__).with_name("schema.sql")


def make_pool(url: str | None = None, max_size: int = 10) -> ConnectionPool:
    # autocommit + dict_row + prepare_threshold=0 are what LangGraph's PostgresSaver needs.
    # statement_timeout bounds every query, so a slow DB becomes a clear error instead of a hang.
    return ConnectionPool(
        url or config.database_url(),
        min_size=1,
        max_size=max_size,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": f"-c statement_timeout={config.db_statement_timeout_ms()} -c TimeZone=Asia/Kolkata",
        },
        open=True,
    )


def init_schema(pool: ConnectionPool) -> None:
    with pool.connection() as conn:
        # ClientCursor uses the simple query protocol, which allows several statements at once.
        ClientCursor(conn).execute(SCHEMA.read_text())
