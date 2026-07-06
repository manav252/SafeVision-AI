from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user
from ..models import Alert, AlertStatus, SafetyEvent, User
from ..schemas import AlertsSummary, EventsSummary, SafetyReport

router = APIRouter()


@router.get("/events-summary", response_model=EventsSummary)
def events_summary(db: Session = Depends(get_db), _: User = Depends(current_user)) -> EventsSummary:
    events = db.query(SafetyEvent).all()
    return summarize_events(events)


@router.get("/alerts-summary", response_model=AlertsSummary)
def alerts_summary(db: Session = Depends(get_db), _: User = Depends(current_user)) -> AlertsSummary:
    alerts = db.query(Alert).all()
    return summarize_alerts(alerts)


@router.get("/safety-report", response_model=SafetyReport)
def safety_report(db: Session = Depends(get_db), _: User = Depends(current_user)) -> SafetyReport:
    events = db.query(SafetyEvent).order_by(SafetyEvent.created_at.desc()).limit(100).all()
    alerts = db.query(Alert).order_by(Alert.created_at.desc()).limit(100).all()
    open_alerts = [alert for alert in alerts if alert.status == AlertStatus.open]
    return SafetyReport(
        generated_at=datetime.utcnow(),
        events=summarize_events(events),
        alerts=summarize_alerts(alerts),
        recent_events=events[:20],
        open_alerts=open_alerts[:20],
    )


def summarize_events(events: list[SafetyEvent]) -> EventsSummary:
    by_severity: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for event in events:
        by_severity[event.severity.value] = by_severity.get(event.severity.value, 0) + 1
        by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
    return EventsSummary(total_events=len(events), by_severity=by_severity, by_type=by_type)


def summarize_alerts(alerts: list[Alert]) -> AlertsSummary:
    by_status: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    active_alerts = 0
    for alert in alerts:
        by_status[alert.status.value] = by_status.get(alert.status.value, 0) + 1
        by_severity[alert.severity.value] = by_severity.get(alert.severity.value, 0) + 1
        if alert.status == AlertStatus.open:
            active_alerts += 1
    return AlertsSummary(
        total_alerts=len(alerts),
        active_alerts=active_alerts,
        by_status=by_status,
        by_severity=by_severity,
    )
