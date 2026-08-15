"""
production/backend/app/api/v1/interaction.py
POST /api/v1/interact  — record a user interaction (like/click/rate)
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..deps import get_current_user
from ...db.session import get_db
from ...db.models import User
from ...schemas.movie import InteractionCreate, InteractionOut
from ...services.interaction_service import InteractionService

router = APIRouter(prefix="/api/v1", tags=["interact"])


@router.post("/interact", response_model=InteractionOut, status_code=201)
def record_interaction(
    payload: InteractionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return InteractionService.record(db, current_user.id, payload)
