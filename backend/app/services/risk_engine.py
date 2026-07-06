from ..models import AlertSeverity


def score_detection(payload: dict) -> tuple[int, list[str]]:
    score = 0
    factors: list[str] = []
    ppe_status = payload.get("ppe_status") or {}
    gas_readings = payload.get("gas_readings") or {}
    zone_status = payload.get("zone_status") or {}
    confidence = float(payload.get("confidence_score") or 0)

    if not ppe_status.get("helmet", True):
        score += 20
        factors.append("missing_helmet")
    if not ppe_status.get("vest", True):
        score += 20
        factors.append("missing_vest")
    if zone_status.get("restricted_zone_breach", False):
        score += 30
        factors.append("restricted_zone_breach")
    if gas_readings.get("methane_lel", 0) >= 10:
        score += 25
        factors.append("methane_elevated")
    if gas_readings.get("co_ppm", 0) >= 35:
        score += 20
        factors.append("co_elevated")
    if gas_readings.get("h2s_ppm", 0) >= 10:
        score += 25
        factors.append("h2s_elevated")
    oxygen = gas_readings.get("oxygen_percent", 20.9)
    if oxygen < 19.5 or oxygen > 23.5:
        score += 25
        factors.append("oxygen_out_of_range")
    if confidence < 0.5:
        factors.append("low_detection_confidence")

    return min(100, score), factors


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
    if "gas" in event_type or "gas" in message or "ch4" in message or "h2s" in message:
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
