from ..models import AlertSeverity


def score_event(event: dict) -> int:
    event_type = str(event.get("event_type", "")).lower()
    message = str(event.get("message", "")).lower()
    score = 0
    if "helmet" in event_type or "helmet" in message:
        score += 20
    if "vest" in event_type or "vest" in message or "ppe" in event_type:
        score += 20
    if "restricted" in event_type or "zone" in event_type or "entry" in message:
        score += 30
    if "gas" in event_type or "ch4" in message or "h2s" in message:
        score += 40
    if "permit" in event_type or "permit" in message:
        score += 25
    return min(100, score)


def classify_severity(score: int) -> AlertSeverity:
    if score >= 85:
        return AlertSeverity.critical
    if score >= 65:
        return AlertSeverity.high
    if score >= 35:
        return AlertSeverity.medium
    return AlertSeverity.low


def risk_level(score: int) -> str:
    return classify_severity(score).value

