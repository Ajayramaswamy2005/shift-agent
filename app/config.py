"""Settings read from environment variables at call time (so tests can override them)."""
import os


def database_url() -> str:
    return os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/shift")


def jwt_secret() -> str:
    return os.getenv("JWT_SECRET", "dev-only-secret-change-me-in-production-32b")


def jwt_ttl_minutes() -> int:
    return int(os.getenv("JWT_TTL_MINUTES", "480"))


def dev_token_issuer_enabled() -> bool:
    # The mock JWT issuer. In production tokens would come from the client's SSO.
    return os.getenv("DEV_TOKEN_ISSUER", "1") == "1"


def db_statement_timeout_ms() -> int:
    return int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "2000"))


def llm_settings() -> dict:
    """Any OpenAI-compatible endpoint. Defaults to Groq's free tier."""
    return {
        "base_url": os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        "model": os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "timeout_s": float(os.getenv("LLM_TIMEOUT_S", "3.0")),
    }
