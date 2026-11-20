"""JWT issue + verify. Identity (company, user, role) comes ONLY from a verified token."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import config

ISSUER = "shift-agent-dev-issuer"
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Identity:
    company_id: str
    user_id: str
    role: str  # "employee" | "manager"

    @property
    def thread_id(self) -> str:
        # One conversation per employee. Derived from the token, never from the request body.
        return f"{self.company_id}:{self.user_id}"


def issue_token(company_id: str, user_id: str, role: str, ttl_minutes: int | None = None) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "sub": user_id,
        "company_id": company_id,
        "role": role,
        "iss": ISSUER,
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes or config.jwt_ttl_minutes()),
    }
    return jwt.encode(claims, config.jwt_secret(), algorithm="HS256")


def decode_token(token: str) -> Identity:
    try:
        claims = jwt.decode(
            token, config.jwt_secret(), algorithms=["HS256"], issuer=ISSUER,
            options={"require": ["sub", "company_id", "role", "exp"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Your session has expired. Please sign in again.")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid sign-in token.")
    if claims["role"] not in ("employee", "manager"):
        raise HTTPException(401, "Invalid sign-in token.")
    return Identity(company_id=claims["company_id"], user_id=claims["sub"], role=claims["role"])


def current_identity(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Identity:
    if creds is None:
        raise HTTPException(401, "Please sign in.")
    return decode_token(creds.credentials)


def employee(ident: Identity = Depends(current_identity)) -> Identity:
    if ident.role != "employee":
        raise HTTPException(403, "This endpoint is for employees.")
    return ident


def manager(ident: Identity = Depends(current_identity)) -> Identity:
    if ident.role != "manager":
        raise HTTPException(403, "This endpoint is for transport managers.")
    return ident
