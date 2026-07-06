from backend.app.models import AlertSeverity, SafetyEvent
from backend.app.services.advisor import generate_advisor_summary
from backend.app.services.risk_engine import classify_severity, score_detection, score_event


def test_risk_engine_scores_detection_context():
    score, factors = score_detection(
        {
            "confidence_score": 0.92,
            "ppe_status": {"helmet": False, "vest": False},
            "gas_readings": {"methane_lel": 15, "co_ppm": 5, "h2s_ppm": 0, "oxygen_percent": 20.9},
            "zone_status": {"restricted_zone_breach": True},
        }
    )

    assert score == 95
    assert factors == ["missing_helmet", "missing_vest", "restricted_zone_breach", "methane_elevated"]
    assert classify_severity(score) == AlertSeverity.critical


def test_risk_engine_scores_event_text():
    score = score_event({"event_type": "gas_permit_risk", "message": "H2S elevated during permit"})

    assert score >= 65
    assert classify_severity(score) == AlertSeverity.high


def test_advisor_logic_empty_and_active_events():
    empty = generate_advisor_summary([])
    assert empty["risk_level"] == "LOW"
    assert empty["confidence"] == 0

    event = SafetyEvent(
        event_type="gas_permit_risk",
        message="Gas elevated during permit",
        risk_score=75,
        severity=AlertSeverity.high,
    )
    summary = generate_advisor_summary([event])

    assert summary["risk_level"] == "HIGH"
    assert "Verify gas reading" in summary["recommended_actions"]
