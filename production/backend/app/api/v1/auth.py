"""
production/backend/app/api/v1/auth.py
POST /api/v1/auth/register
POST /api/v1/auth/login
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...db.session import get_db
from ...schemas.user import TokenOut, UserLogin, UserRegister
from ...services.user_service import UserService

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=TokenOut, status_code=201)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    return UserService.register(db, payload)


@router.post("/login", response_model=TokenOut)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    return UserService.login(db, payload)
