from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from auth import create_access_token, get_current_user, hash_password, verify_password
from database import User, get_db

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    password: str
    openrouter_api_key: str = ""
    metacentrum_api_key: str = ""
    llm_provider: str = "openrouter"

    @field_validator("username")
    @classmethod
    def _clean_username(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if len(v) > 40:
            raise ValueError("Username must be at most 40 characters.")
        return v

    @field_validator("password")
    @classmethod
    def _check_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter_by(username=body.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken.",
        )
    user = User(
        username=body.username,
        hashed_password=hash_password(body.password),
        openrouter_api_key=body.openrouter_api_key.strip() or None,
        metacentrum_api_key=body.metacentrum_api_key.strip() or None,
        llm_provider=body.llm_provider if body.llm_provider in ("openrouter", "metacentrum") else "openrouter",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        username=user.username,
    )


@router.post("/login", response_model=TokenResponse)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter_by(username=form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        username=user.username,
    )


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "ok": True,
        "username": current_user.username,
        "has_openrouter_key": bool(current_user.openrouter_api_key),
        "has_metacentrum_key": bool(current_user.metacentrum_api_key),
        "has_api_key": bool(current_user.openrouter_api_key or current_user.metacentrum_api_key),
        "llm_provider": getattr(current_user, "llm_provider", None) or "openrouter",
    }


class UpdateApiKeyRequest(BaseModel):
    llm_provider: str
    openrouter_api_key: str = ""
    metacentrum_api_key: str = ""


@router.put("/api-key", status_code=200)
def update_api_key(
    body: UpdateApiKeyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Store or update the user's LLM provider and API keys."""
    if body.llm_provider in ("openrouter", "metacentrum"):
        current_user.llm_provider = body.llm_provider
    current_user.openrouter_api_key = body.openrouter_api_key.strip() or current_user.openrouter_api_key
    current_user.metacentrum_api_key = body.metacentrum_api_key.strip() or current_user.metacentrum_api_key
    db.commit()
    provider = getattr(current_user, "llm_provider", None) or "openrouter"
    has_key = bool(current_user.openrouter_api_key if provider == "openrouter" else current_user.metacentrum_api_key)
    return {"ok": True, "has_api_key": has_key, "llm_provider": provider}
