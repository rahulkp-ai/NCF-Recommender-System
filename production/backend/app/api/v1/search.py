"""
production/backend/app/api/v1/search.py
GET /api/v1/search?q=<query>
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ...db.session import get_db
from ...schemas.movie import SearchResponse
from ...services.search_service import SearchService

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=50),
    db: Session = Depends(get_db),
):
    return SearchService.search(db, q, limit=limit)
