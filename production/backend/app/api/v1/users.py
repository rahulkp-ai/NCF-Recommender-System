"""
production/backend/app/api/v1/users.py
GET /api/v1/users/profile  — returns current user profile
"""

from fastapi import APIRouter, Depends

from ...db.models import User
from ...schemas.user import UserOut
from ..deps import get_current_user

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.get("/profile", response_model=UserOut)
def get_profile(current_user: User = Depends(get_current_user)):
    return current_user
