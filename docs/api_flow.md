# SafeVision AI API Flow

This walkthrough shows the backend API path for authentication, detection intake, events, alerts, reports, and dashboard summaries.

Start the stack first:

```bash
docker compose up --build
```

API docs:

```text
http://localhost:8000/docs
```

## 1. Register a Demo User

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "demo@safevision.ai",
    "full_name": "SafeVision Demo User",
    "password": "ChangeMe123",
    "role": "operator"
  }'
```

If the user already exists, continue to login.

## 2. Login

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo@safevision.ai&password=ChangeMe123" \
  | python -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
```

## 3. Submit a Detection

```bash
curl -X POST http://localhost:8000/api/v1/detection/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detection_type": "restricted_zone_breach",
    "confidence_score": 0.82,
    "ppe_status": {
      "helmet": false,
      "vest": true
    },
    "gas_readings": {
      "methane_lel": 12,
      "co_ppm": 4,
      "h2s_ppm": 0,
      "oxygen_percent": 20.9
    },
    "zone_status": {
      "zone_name": "Zone A",
      "restricted_zone_breach": true
    },
    "metadata": {
      "source": "api_demo",
      "frame": 42
    }
  }'
```

The detection API calculates risk factors, creates a safety event, and creates an alert when the risk threshold is met.

## 4. List Events

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/events/
```

## 5. List Alerts

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/alerts/
```

## 6. Dashboard Summary

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/dashboard/summary
```

## 7. Export-Ready Safety Report JSON

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/reports/safety-report
```

## 8. Verify In PostgreSQL

```bash
docker compose exec postgres psql -U safevision -d safevision
```

```sql
SELECT zone_name, event_type, severity, risk_score, created_at
FROM safety_events
ORDER BY created_at DESC;

SELECT title, severity, status, created_at
FROM alerts
ORDER BY created_at DESC;
```

## Notes

- Uploaded video files are not stored in PostgreSQL.
- Detection metadata is stored in `safety_events.metadata_json`.
- Reports are generated from stored events and alerts rather than stored as separate report files.
- The API can be explored interactively through Swagger at `/docs`.
