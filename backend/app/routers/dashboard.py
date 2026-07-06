from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user
from ..models import Alert, AlertStatus, SafetyEvent, User
from ..schemas import DashboardSummary
from ..services.heatmap import build_heatmap

router = APIRouter()


@router.get("/summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db), _: User = Depends(current_user)) -> DashboardSummary:
    events = db.query(SafetyEvent).order_by(SafetyEvent.created_at.desc()).limit(250).all()
    active_alerts = db.query(Alert).filter(Alert.status == AlertStatus.open).count()
    return DashboardSummary(
        total_events=len(events),
        active_alerts=active_alerts,
        risk_distribution=build_risk_distribution(events),
        recent_incidents=events[:10],
        heatmap_summary=build_heatmap(events),
    )


def build_risk_distribution(events: list[SafetyEvent]) -> dict[str, int]:
    distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for event in events:
        distribution[event.severity.value] = distribution.get(event.severity.value, 0) + 1
    return distribution
