from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.schemas import AuthUser, LoginRequest, RegisterRequest, TokenResponse
from core.database import get_db
from core.models import User
from core.security import create_access_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    username = request.username.strip()
    email = request.email.strip().lower()

    existing = db.scalar(select(User).where(or_(User.username == username, User.email == email)))
    if existing:
        field = "username" if existing.username == username else "email"
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"{field} already exists.")

    user = User(username=username, email=email, password_hash=hash_password(request.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        existing = db.scalar(select(User).where(or_(User.username == username, User.email == email)))
        if existing is not None:
            field = "username" if existing.username == username else "email"
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"{field} already exists.") from exc
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="username or email already exists.",
        ) from exc
    db.refresh(user)

    return TokenResponse(access_token=create_access_token(user), user=AuthUser.model_validate(user))


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    identifier = request.identifier.strip()
    user = db.scalar(
        select(User).where(or_(User.username == identifier, User.email == identifier.lower()))
    )
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username/email or password.")

    return TokenResponse(access_token=create_access_token(user), user=AuthUser.model_validate(user))


@router.get("/me", response_model=AuthUser)
def me(current_user: User = Depends(get_current_user)) -> AuthUser:
    return AuthUser.model_validate(current_user)
