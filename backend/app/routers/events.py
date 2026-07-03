from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user
from ..models import Alert, SafetyEvent, User
from ..schemas import SafetyEventCreate, SafetyEventRead
from ..services.risk_engine import classify_severity, score_event

router = APIRouter()


@router.post("/", response_model=SafetyEventRead)
def create_event(payload: SafetyEventCreate, db: Session = Depends(get_db), _: User = Depends(current_user)) -> SafetyEvent:
    risk_score = score_event(payload.model_dump())
    event = SafetyEvent(
        **payload.model_dump(),
        risk_score=risk_score,
        severity=classify_severity(risk_score),
    )
    db.add(event)
    db.flush()
    if risk_score >= 35:
        db.add(Alert(event_id=event.id, title=payload.message[:150], severity=event.severity))
    db.commit()
    db.refresh(event)
    return event


@router.get("/", response_model=list[SafetyEventRead])
def list_events(db: Session = Depends(get_db), _: User = Depends(current_user)) -> list[SafetyEvent]:
    return db.query(SafetyEvent).order_by(SafetyEvent.created_at.desc()).limit(100).all()

