"""
production/backend/app/services/user_service.py
Business logic for user registration and authentication.
"""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..core.security import create_access_token, hash_password, verify_password
from ..db.repository import UserRepository
from ..schemas.user import TokenOut, UserLogin, UserOut, UserRegister


class UserService:
    @staticmethod
    def register(db: Session, payload: UserRegister) -> TokenOut:
        if UserRepository.get_by_username(db, payload.username):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already taken",
            )
        if UserRepository.get_by_email(db, payload.email):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )
        user = UserRepository.create(
            db,
            username=payload.username,
            email=payload.email,
            password_hash=hash_password(payload.password),
            age_group=payload.age_group,
            gender=payload.gender,
        )
        token = create_access_token({"sub": str(user.id)})
        return TokenOut(access_token=token, user=UserOut.model_validate(user))

    @staticmethod
    def login(db: Session, payload: UserLogin) -> TokenOut:
        user = UserRepository.get_by_username(db, payload.username)
        if not user or not verify_password(payload.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )
        token = create_access_token({"sub": str(user.id)})
        return TokenOut(access_token=token, user=UserOut.model_validate(user))
