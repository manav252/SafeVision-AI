def test_health_endpoint(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_auth_register_and_login(client):
    payload = {
        "email": "operator@safevision.ai",
        "full_name": "Plant Operator",
        "password": "ChangeMe123",
        "role": "operator",
    }

    register = client.post("/api/v1/auth/register", json=payload)
    assert register.status_code == 200
    assert register.json()["email"] == payload["email"]

    login = client.post(
        "/api/v1/auth/login",
        data={"username": payload["email"], "password": payload["password"]},
    )
    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    assert login.json()["access_token"]


def test_events_api_creates_event_and_alert(client, auth_headers):
    payload = {
        "zone_name": "Zone A",
        "event_type": "restricted_zone_breach",
        "message": "Worker entered restricted zone with gas warning",
    }

    created = client.post("/api/v1/events/", json=payload, headers=auth_headers)
    assert created.status_code == 200
    assert created.json()["risk_score"] >= 35

    events = client.get("/api/v1/events/", headers=auth_headers)
    assert events.status_code == 200
    assert len(events.json()) == 1


def test_alerts_api_lists_and_acknowledges_alert(client, auth_headers):
    client.post(
        "/api/v1/events/",
        json={
            "zone_name": "Zone A",
            "event_type": "gas_permit_risk",
            "message": "Gas elevated during active permit",
        },
        headers=auth_headers,
    )

    alerts = client.get("/api/v1/alerts/", headers=auth_headers)
    assert alerts.status_code == 200
    alert_id = alerts.json()[0]["id"]

    acknowledged = client.patch(f"/api/v1/alerts/{alert_id}/acknowledge", headers=auth_headers)
    assert acknowledged.status_code == 200
    assert acknowledged.json()["status"] == "ACKNOWLEDGED"


def test_detection_api_records_detection_event(client, auth_headers):
    payload = {
        "detection_type": "person",
        "confidence_score": 0.87,
        "worker_id": "WKR-204",
        "ppe_status": {"helmet": False, "vest": True},
        "gas_readings": {"methane_lel": 12, "co_ppm": 10, "h2s_ppm": 0, "oxygen_percent": 20.9},
        "zone_status": {"zone_name": "Zone B", "restricted_zone_breach": True},
        "metadata": {"bbox": [10, 20, 100, 220]},
    }

    response = client.post("/api/v1/detection/", json=payload, headers=auth_headers)
    body = response.json()

    assert response.status_code == 200
    assert body["calculated_risk_score"] >= 35
    assert body["alert_created"] is True
    assert "missing_helmet" in body["risk_factors"]
    assert body["event"]["zone_name"] == "Zone B"


def test_dashboard_summary(client, auth_headers):
    client.post(
        "/api/v1/detection/",
        json={
            "confidence_score": 0.91,
            "ppe_status": {"helmet": False, "vest": False},
            "zone_status": {"zone_name": "Zone C", "restricted_zone_breach": False},
        },
        headers=auth_headers,
    )

    response = client.get("/api/v1/dashboard/summary", headers=auth_headers)
    body = response.json()

    assert response.status_code == 200
    assert body["total_events"] == 1
    assert body["active_alerts"] == 1
    assert body["recent_incidents"][0]["zone_name"] == "Zone C"
    assert body["heatmap_summary"]["zones"]


def test_reports_api_returns_export_ready_json(client, auth_headers):
    client.post(
        "/api/v1/events/",
        json={"zone_name": "Zone A", "event_type": "gas_permit_risk", "message": "Gas elevated during active permit"},
        headers=auth_headers,
    )

    response = client.get("/api/v1/reports/safety-report", headers=auth_headers)
    body = response.json()

    assert response.status_code == 200
    assert body["events"]["total_events"] == 1
    assert body["alerts"]["total_alerts"] == 1
    assert body["recent_events"]
