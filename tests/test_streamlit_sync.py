from safevision_api_client import build_detection_payload


def test_streamlit_violation_maps_to_detection_payload():
    payload = build_detection_payload(
        {
            "frame": 42,
            "timestamp": "1.68s",
            "violation_type": "restricted_zone_breach",
            "severity": "HIGH",
            "risk_points": 40,
            "zone_name": "Zone A",
            "message": "Worker entered Zone A",
        },
        {
            "readings": {
                "methane_lel": 12,
                "co_ppm": 4,
                "h2s_ppm": 0,
                "oxygen_pct": 20.9,
            }
        },
    )

    assert payload["detection_type"] == "restricted_zone_breach"
    assert payload["zone_status"]["zone_name"] == "Zone A"
    assert payload["zone_status"]["restricted_zone_breach"] is True
    assert payload["gas_readings"]["methane_lel"] == 12
    assert payload["metadata"]["source"] == "streamlit"


def test_ppe_violation_maps_to_ppe_status():
    helmet_payload = build_detection_payload({"violation_type": "no_helmet"})
    vest_payload = build_detection_payload({"violation_type": "no_vest"})

    assert helmet_payload["ppe_status"]["helmet"] is False
    assert helmet_payload["ppe_status"]["vest"] is True
    assert vest_payload["ppe_status"]["helmet"] is True
    assert vest_payload["ppe_status"]["vest"] is False
