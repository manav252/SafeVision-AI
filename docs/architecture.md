# SafeVision AI Architecture

SafeVision AI is structured as a portfolio-ready industrial safety prototype with a React/Vite website, Streamlit live dashboard, YOLO/OpenCV detection layer, risk engine, FastAPI backend, and PostgreSQL persistence.

## Current System Architecture

```mermaid
flowchart TD
    website["React/Vite Website"]
    streamlit["Streamlit Dashboard"]
    vision["YOLO/OpenCV Detection"]
    risk["Risk Engine"]
    advisor["AI Safety Advisor"]
    api["FastAPI Backend"]
    db["PostgreSQL"]
    outputs["Dashboard / Reports / Alerts"]

    website --> streamlit
    streamlit --> vision
    vision --> risk
    risk --> advisor
    risk --> api
    api --> db
    db --> outputs
```

## Component Responsibilities

| Component | Responsibility |
| --- | --- |
| `src/` | React/Vite website used as the project landing page and Vercel deploy target. |
| `app.py` | Streamlit operations dashboard for video upload, camera setup, zone drawing, monitoring, advisor UI, heatmap, and report drafting. |
| `detector.py` | YOLOv8/Ultralytics model loading, frame inference, PPE class parsing, and fallback detection behavior. |
| `risk_engine.py` | Weighted risk scoring and safety event generation for PPE, restricted zones, gas context, permits, equipment, shift state, and emergency context. |
| `safevision_api_client.py` | Streamlit-to-FastAPI client that syncs newly detected violation rows into the backend. |
| `backend/app/` | FastAPI backend with authentication, events, alerts, detection intake, reports, dashboard summary, and heatmap APIs. |
| `backend/app/models.py` | SQLAlchemy models for users, cameras, safety events, alerts, and plant signals. |
| PostgreSQL | Stores users, cameras, safety events, alerts, plant signals, detection metadata embedded in events, and report source data. |
| Docker Compose | Runs PostgreSQL, FastAPI, and Streamlit together for local full-stack demos. |

## Detection and Persistence Flow

```mermaid
flowchart TD
    upload["Video Upload or Demo Feed"]
    frames["Frame Extraction"]
    yolo["YOLO/OpenCV Detection"]
    zone["Restricted Zone Check"]
    context["Plant Context"]
    risk["Risk Engine"]
    advisor["AI Safety Advisor"]
    client["Streamlit API Client"]
    backend["FastAPI Detection API"]
    db["PostgreSQL"]
    dashboard["Dashboard / Reports / Alerts"]

    upload --> frames
    frames --> yolo
    yolo --> zone
    context --> risk
    zone --> risk
    risk --> advisor
    risk --> client
    client --> backend
    backend --> db
    db --> dashboard
    advisor --> dashboard
```

## Backend API Surface

| API Area | Purpose |
| --- | --- |
| `/api/v1/auth` | Register users and issue JWT access tokens. |
| `/api/v1/events` | Create and list safety events. |
| `/api/v1/alerts` | List and acknowledge generated alerts. |
| `/api/v1/detection` | Accept detection metadata, PPE state, gas readings, zone state, confidence, and calculated risk context. |
| `/api/v1/reports` | Generate export-ready report JSON from stored events and alerts. |
| `/api/v1/dashboard` | Return dashboard totals, active alerts, risk distribution, recent incidents, and heatmap summary. |
| `/api/v1/heatmap` | Return zone-level heatmap data. |
| `/api/v1/health` | Report API health. |

## Data Model Notes

The current implementation persists detection results as `safety_events` with detailed detection payloads in `metadata_json`. Alerts are generated from safety events. Reports are generated as export-ready JSON from stored events and alerts rather than saved as separate report files.

```mermaid
erDiagram
    USERS {
        uuid id PK
        string email
        string full_name
        string role
        boolean is_active
        datetime created_at
    }

    CAMERAS {
        uuid id PK
        string name
        text stream_url
        string zone_name
        string status
        json restricted_zone
        datetime created_at
        datetime updated_at
    }

    SAFETY_EVENTS {
        uuid id PK
        uuid camera_id FK
        string zone_name
        string event_type
        string severity
        text message
        int risk_score
        string worker_id
        text evidence_uri
        json metadata_json
        datetime created_at
    }

    ALERTS {
        uuid id PK
        uuid event_id FK
        string title
        string severity
        string status
        string assigned_to
        text response_notes
        datetime created_at
        datetime updated_at
    }

    PLANT_SIGNALS {
        uuid id PK
        string zone_name
        float methane_lel
        float co_ppm
        float h2s_ppm
        float oxygen_percent
        string permit_type
        string equipment_status
        string shift_status
        datetime created_at
    }

    CAMERAS ||--o{ SAFETY_EVENTS : records
    SAFETY_EVENTS ||--o| ALERTS : creates
    PLANT_SIGNALS }o--o{ SAFETY_EVENTS : contextualizes
```

## Deployment View

```mermaid
flowchart TD
    browser["Browser"]
    vercel["React Website on Vercel"]

    subgraph compose["Docker Compose"]
        streamlit["Streamlit Dashboard"]
        api["FastAPI Backend"]
        db["PostgreSQL Database"]
    end

    browser --> vercel
    browser --> streamlit
    browser --> api
    streamlit --> api
    api --> db
```

## Current Limitations

- Gas readings and plant signals are demo inputs unless connected to real plant systems.
- The PPE model is a Roboflow-exported pretrained YOLOv8 model and has not been fine-tuned in this repository.
- Formal model evaluation metrics are not claimed in this project.
- PostgreSQL stores backend event and alert data; uploaded video files remain file-based under `outputs/uploads/` for local runs.
- The system is a prototype and is not certified for production industrial safety decisions.

## Future Work

- Fine-tune and evaluate PPE detection on representative industrial datasets.
- Connect live gas detector, SCADA, historian, and permit-to-work APIs.
- Add cloud object storage for video/evidence retention.
- Add Alembic migrations for production-grade database evolution.
- Add role-based review workflows and real notification channels.
