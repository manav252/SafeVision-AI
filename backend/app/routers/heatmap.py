from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user
from ..models import SafetyEvent, User
from ..services.heatmap import build_heatmap

router = APIRouter()


@router.get("/")
def get_heatmap(db: Session = Depends(get_db), _: User = Depends(current_user)) -> dict:
    events = db.query(SafetyEvent).order_by(SafetyEvent.created_at.desc()).limit(250).all()
    return build_heatmap(events)

