from ..models import SafetyEvent


def generate_advisor_summary(events: list[SafetyEvent]) -> dict:
    if not events:
        return {
            "risk_level": "LOW",
            "risk_score": 0,
            "summary": "No active safety events. Continue monitoring configured cameras.",
            "recommended_actions": ["Keep CCTV feeds online", "Maintain gas sensor polling", "Review permits before shift handover"],
            "confidence": 0,
        }

    max_score = max(event.risk_score for event in events)
    recent_types = {event.event_type for event in events[:10]}
    actions = []
    if any("gas" in item.lower() for item in recent_types):
        actions.extend(["Verify gas reading", "Increase ventilation", "Pause hot work"])
    if any("zone" in item.lower() or "entry" in item.lower() for item in recent_types):
        actions.extend(["Clear restricted zone", "Notify area supervisor", "Preserve CCTV evidence"])
    if any("ppe" in item.lower() or "helmet" in item.lower() or "vest" in item.lower() for item in recent_types):
        actions.extend(["Recheck PPE compliance", "Issue worker warning"])
    actions = list(dict.fromkeys(actions))[:5] or ["Notify safety administrator", "Review latest events"]
    return {
        "risk_level": "CRITICAL" if max_score >= 85 else "HIGH" if max_score >= 65 else "MEDIUM" if max_score >= 35 else "LOW",
        "risk_score": max_score,
        "summary": "SafeVision AI detected correlated safety signals across CCTV, plant context, and operating permits.",
        "recommended_actions": actions,
        "confidence": min(98, 55 + len(events[:10]) * 4),
    }

