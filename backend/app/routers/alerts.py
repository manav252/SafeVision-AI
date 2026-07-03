from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user, require_roles
from ..models import Alert, AlertStatus, User, UserRole
from ..schemas import AlertRead

router = APIRouter()


@router.get("/", response_model=list[AlertRead])
def list_alerts(status: AlertStatus | None = None, db: Session = Depends(get_db), _: User = Depends(current_user)) -> list[Alert]:
    query = db.query(Alert)
    if status:
        query = query.filter(Alert.status == status)
    return query.order_by(Alert.created_at.desc()).limit(100).all()


@router.patch("/{alert_id}/acknowledge", response_model=AlertRead)
def acknowledge_alert(
    alert_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(UserRole.admin, UserRole.safety_officer, UserRole.operator)),
) -> Alert:
    alert = db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = AlertStatus.acknowledged
    alert.assigned_to = user.email
    db.commit()
    db.refresh(alert)
    return alert

