# SafeVision AI Demo Script

Use this script for a short portfolio, interview, or class presentation demo.

## 60-Second Demo

1. Start the full stack.

   ```bash
   docker compose up --build
   ```

2. Open the Streamlit dashboard.

   ```text
   http://localhost:8501
   ```

3. Open the FastAPI docs in another tab.

   ```text
   http://localhost:8000/docs
   ```

4. Explain the architecture.

   ```text
   React/Vite website -> Streamlit dashboard -> YOLO/OpenCV detection -> Risk Engine -> FastAPI -> PostgreSQL -> Dashboard / Reports / Alerts
   ```

5. In Streamlit, click **Launch Live Dashboard**.

6. Upload a CCTV clip or use the industrial CCTV demo mode.

7. Assign a plant zone and configure plant context such as gas + permit risk.

8. Start monitoring.

9. Point out:

   - worker/PPE/restricted-zone detection
   - risk score
   - AI Safety Advisor
   - recent safety events
   - heatmap/report tabs

10. Show the backend proof.

    ```bash
    docker compose exec postgres psql -U safevision -d safevision
    ```

    ```sql
    SELECT zone_name, event_type, severity, risk_score, created_at
    FROM safety_events
    ORDER BY created_at DESC;
    ```

## Suggested Talk Track

SafeVision AI is an industrial safety intelligence prototype. It combines CCTV-based PPE and restricted-zone detection with operational plant context such as gas readings, permits, equipment state, shift handover, and compliance checklist state.

The Streamlit dashboard handles the live demo experience, while FastAPI provides backend APIs for authentication, events, alerts, detections, reports, dashboard summaries, and heatmap data. PostgreSQL stores the backend safety events and alerts so the system is not just a visual demo; it has a real persistence layer.

The model component is YOLOv8-based and uses a Roboflow-exported pretrained PPE model. The project does not claim formal production metrics yet. It is a research and portfolio prototype, with future work focused on real plant data, model evaluation, and sensor integrations.

## What To Emphasize

- This is a full-stack AI safety prototype, not only a UI.
- The backend runs with FastAPI and PostgreSQL.
- Docker Compose starts PostgreSQL, FastAPI, and Streamlit together.
- Reports and dashboard summaries are generated from stored safety events and alerts.
- The model card is honest about current limitations and missing formal evaluation metrics.

## What Not To Claim

- Do not claim production certification.
- Do not claim real deployed plant sensor integration.
- Do not claim accuracy, precision, recall, F1-score, or mAP until formal evaluation exists.
- Do not claim the PPE model was trained from scratch in this repository.
