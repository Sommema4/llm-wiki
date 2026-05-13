"""
Auth utilities: password hashing (bcrypt) and JWT creation/validation.
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database import User, get_db

# ── Secret key ────────────────────────────────────────────────────────────────
# For a real deployment put this in .env.  For local dev a random default is fine.
_SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-use-a-long-random-string")
_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

# ── Password context ──────────────────────────────────────────────────────────
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── OAuth2 bearer ─────────────────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": user_id, "exp": expire}, _SECRET_KEY, algorithm=_ALGORITHM)


def _decode_token(token: str) -> str:
    """Return user_id from a valid token, raise 401 otherwise."""
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise JWTError("missing sub")
        return user_id
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependency ─────────────────────────────────────────────────────────

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    user_id = _decode_token(token)
    user = db.query(User).filter_by(id=user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    return user


def get_user_llm_creds(user: User) -> Tuple[Optional[str], Optional[str], str, str]:
    """Return (api_key, base_url, default_model, chat_model) for the user's chosen provider."""
    from config import get_settings
    s = get_settings()
    provider = getattr(user, "llm_provider", None) or "openrouter"
    if provider == "metacentrum":
        return user.metacentrum_api_key, s.metacentrum_base_url, s.metacentrum_default_model, s.metacentrum_chat_model
    return user.openrouter_api_key, None, s.default_model, s.chat_model
