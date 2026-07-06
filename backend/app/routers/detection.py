from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user
from ..models import Alert, SafetyEvent, User
from ..schemas import DetectionCreate, DetectionRead
from ..services.risk_engine import classify_severity, score_detection

router = APIRouter()


@router.post("/", response_model=DetectionRead)
def create_detection(
    payload: DetectionCreate,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> DetectionRead:
    payload_dict = payload.model_dump()
    risk_score, factors = score_detection(payload_dict)
    zone_name = payload.zone_status.zone_name
    message = build_detection_message(payload, factors)
    event = SafetyEvent(
        camera_id=payload.camera_id,
        zone_name=zone_name,
        event_type="detection",
        message=message,
        worker_id=payload.worker_id,
        risk_score=risk_score,
        severity=classify_severity(risk_score),
        metadata_json={
            "detection_type": payload.detection_type,
            "confidence_score": payload.confidence_score,
            "ppe_status": payload.ppe_status.model_dump(),
            "gas_readings": payload.gas_readings.model_dump(),
            "zone_status": payload.zone_status.model_dump(),
            "risk_factors": factors,
            "metadata": payload.metadata or {},
        },
    )
    db.add(event)
    db.flush()
    alert_created = risk_score >= 35
    if alert_created:
        db.add(Alert(event_id=event.id, title=message[:150], severity=event.severity))
    db.commit()
    db.refresh(event)
    return DetectionRead(event=event, calculated_risk_score=risk_score, alert_created=alert_created, risk_factors=factors)


def build_detection_message(payload: DetectionCreate, factors: list[str]) -> str:
    subject = payload.worker_id or payload.detection_type
    if not factors:
        return f"Detection recorded for {subject} with no active safety risk factors."
    return f"Detection recorded for {subject}: {', '.join(factors)}."
