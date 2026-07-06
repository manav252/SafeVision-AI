# SafeVision AI API Documentation

Base URL:

```text
http://localhost:8000/api/v1
```

Interactive documentation is available at:

```text
http://localhost:8000/docs
```

## Authentication

### Register User

`POST /auth/register`

```json
{
  "email": "admin@safevision.ai",
  "full_name": "Plant Safety Admin",
  "password": "ChangeMe123",
  "role": "admin"
}
```

### Login

`POST /auth/login`

Form fields:

```text
username=admin@safevision.ai
password=ChangeMe123
```

Response:

```json
{
  "access_token": "jwt-token",
  "token_type": "bearer"
}
```

## Cameras

### Create Camera

`POST /cameras/`

```json
{
  "name": "CCTV-1",
  "stream_url": "rtsp://camera.local/zone-a",
  "zone_name": "Zone A",
  "restricted_zone": {
    "type": "polygon",
    "points": [[120, 260], [420, 260], [420, 520], [120, 520]]
  }
}
```

### List Cameras

`GET /cameras/`

## Safety Events

### Create Safety Event

`POST /events/`

```json
{
  "camera_id": "00000000-0000-0000-0000-000000000000",
  "zone_name": "Zone A",
  "event_type": "no_helmet",
  "message": "Helmet not detected for worker in Zone A",
  "worker_id": "WKR-204",
  "evidence_uri": "outputs/evidence/frame_001.jpg",
  "metadata_json": {
    "bbox": [180, 90, 410, 620],
    "confidence": 0.86,
    "permit": "Maintenance Permit"
  }
}
```

## Alerts

`GET /alerts/`

`PATCH /alerts/{alert_id}/acknowledge`

## Detection

### Create Detection Event

`POST /detection/`

```json
{
  "camera_id": "00000000-0000-0000-0000-000000000000",
  "worker_id": "WKR-204",
  "detection_type": "person",
  "confidence_score": 0.87,
  "ppe_status": {
    "helmet": false,
    "vest": true
  },
  "gas_readings": {
    "methane_lel": 12,
    "co_ppm": 10,
    "h2s_ppm": 0,
    "oxygen_percent": 20.9
  },
  "zone_status": {
    "zone_name": "Zone B",
    "restricted_zone_breach": true
  },
  "metadata": {
    "bbox": [10, 20, 100, 220]
  }
}
```

The endpoint calculates a risk score, creates a safety event, and opens an alert when the calculated score is medium or higher.

## Reports

`GET /reports/events-summary`

`GET /reports/alerts-summary`

`GET /reports/safety-report`

The safety report response is export-ready JSON containing event counts, alert counts, recent events, and open alerts.

## Dashboard

`GET /dashboard/summary`

Returns total events, active alerts, risk distribution, recent incidents, and heatmap summary.

## Heatmap

`GET /heatmap/`

```json
{
  "zones": [
    {
      "zone": "Zone A",
      "risk_score": 86,
      "risk_level": "CRITICAL",
      "event_count": 4,
      "factors": ["restricted_zone_entry", "gas", "permit"]
    }
  ]
}
```

## AI Safety Advisor

`GET /advisor/summary`

```json
{
  "risk_level": "HIGH",
  "risk_score": 72,
  "summary": "SafeVision AI detected correlated safety signals across CCTV, plant context, and operating permits.",
  "recommended_actions": [
    "Verify gas reading",
    "Increase ventilation",
    "Pause hot work"
  ],
  "confidence": 83
}
```
