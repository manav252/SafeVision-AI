from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user
from ..models import SafetyEvent, User
from ..services.advisor import generate_advisor_summary

router = APIRouter()


@router.get("/summary")
def advisor_summary(db: Session = Depends(get_db), _: User = Depends(current_user)) -> dict:
    events = db.query(SafetyEvent).order_by(SafetyEvent.created_at.desc()).limit(50).all()
    return generate_advisor_summary(events)

