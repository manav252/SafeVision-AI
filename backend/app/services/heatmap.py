from collections import defaultdict

from ..models import SafetyEvent


def build_heatmap(events: list[SafetyEvent]) -> dict:
    zones: dict[str, dict] = defaultdict(lambda: {"risk_score": 0, "events": 0, "factors": set()})
    for event in events:
        zone = event.zone_name or "Plant"
        zones[zone]["risk_score"] = max(zones[zone]["risk_score"], event.risk_score)
        zones[zone]["events"] += 1
        zones[zone]["factors"].add(event.event_type)

    result = []
    for zone, data in zones.items():
        score = min(100, data["risk_score"] + min(15, data["events"]))
        result.append(
            {
                "zone": zone,
                "risk_score": score,
                "risk_level": "CRITICAL" if score >= 85 else "HIGH" if score >= 65 else "MEDIUM" if score >= 35 else "LOW",
                "event_count": data["events"],
                "factors": sorted(data["factors"]),
            }
        )
    return {"zones": sorted(result, key=lambda item: item["risk_score"], reverse=True)}

