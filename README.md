# SafeVision AI

SafeVision AI is an industrial safety intelligence platform that combines CCTV analytics with plant context such as gas readings, work permits, equipment status, shift handover notes, restricted zones, and compliance checklist state.

The repository includes two demo surfaces:

- **Website demo:** React + Vite landing page for Vercel deployment.
- **Live operations dashboard:** Streamlit app with video upload, zone drawing, YOLO/OpenCV detection, risk fusion, AI Safety Advisor, heatmap, evidence capture, and incident logs.

## What It Solves

Industrial sites often run CCTV, gas detectors, permit systems, and compliance workflows separately. SafeVision AI fuses those signals so a safety officer can see compound risk before it becomes an incident.

Example:

```text
Worker near restricted zone
+ PPE warning
+ elevated gas
+ active maintenance permit
= high-priority supervisor action
```

## Core Features

- Multi-camera CCTV manager for uploaded plant feeds.
- YOLOv8/OpenCV based person and PPE detection.
- Custom PPE model support at `models/ppe_yolov8.pt`.
- Fallback PPE estimation when a custom model is unavailable.
- Freehand and preset restricted-zone monitoring.
- Plant signal inputs for gas, permits, equipment, shift handover, compliance, and emergency state.
- Weighted risk score from 0 to 100.
- AI Safety Advisor with reasoned recommendations.
- Explain-this-alert workflow for demo explainability.
- Plant Risk Heatmap showing zone-level risk.
- Evidence screenshots and CSV logs.
- FastAPI backend scaffold with PostgreSQL schema for enterprise extension.
- Docker and GitHub Actions support.

## Tech Stack

**Frontend website:** React, Vite, Tailwind CSS, Framer Motion  
**Live dashboard:** Streamlit, streamlit-drawable-canvas, OpenCV, Ultralytics YOLOv8, NumPy, Pandas, Pillow  
**Backend scaffold:** FastAPI, SQLAlchemy, PostgreSQL, JWT auth  
**Deployment:** Vercel for the website, Docker/Streamlit for the live dashboard

## Repository Structure

```text
SafeVision-AI/
├── src/                       # React/Vite landing page
├── app.py                     # Original Streamlit dashboard
├── detector.py                # YOLO loading, inference, PPE fallback logic
├── risk_engine.py             # Risk score and safety event generation
├── utils.py                   # Drawing, geometry, evidence, CSV utilities
├── backend/                   # FastAPI enterprise API scaffold
├── database/schema.sql        # PostgreSQL schema
├── docs/                      # Architecture, API, deployment docs
├── assets/                    # Architecture diagram and landing assets
├── models/                    # YOLO model files
├── sample_videos/             # Demo CCTV footage
├── requirements.txt           # Streamlit/Python dependencies
├── package.json               # React/Vite dependencies
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
└── vercel.json
```

## Run Website Locally

```bash
npm install
npm run dev
```

Open the Vite URL shown in the terminal.

## Run Live Dashboard Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Website To Vercel

This repo is already configured for Vercel.

- Framework preset: `Vite`
- Build command: `npm run build`
- Output directory: `dist`

Import the GitHub repo into Vercel and deploy. The landing page can link users into the live dashboard demo flow.

## Docker Demo

```bash
cp .env.example .env
docker compose up --build
```

Open:

- Streamlit dashboard: `http://localhost:8501`
- FastAPI backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Demo Flow

1. Open the SafeVision AI landing page.
2. Click **Launch Live Dashboard**.
3. Upload one or more CCTV clips or use the industrial demo feed.
4. Draw restricted zones for plant areas.
5. Select plant context such as elevated gas and active permit.
6. Start monitoring.
7. Show live detection, risk score, recent safety events, AI Safety Advisor, heatmap, and incident report.

## Production Notes

- Replace fallback PPE estimation with a site-trained model at `models/ppe_yolov8.pt`.
- Connect gas readings to PLC, SCADA, MQTT, OPC-UA, or historian APIs.
- Store evidence in object storage for production.
- Use managed PostgreSQL and secure JWT configuration.
- Add real alert channels such as email, SMS, WhatsApp, Teams, or plant siren integration.

## Resume Bullets

- Built an industrial safety intelligence platform combining YOLOv8 computer vision with gas, permit, equipment, shift, and compliance context.
- Implemented restricted-zone monitoring, PPE violation detection, weighted risk scoring, and explainable safety recommendations.
- Designed a Vercel-ready React landing page and a Streamlit operations dashboard with evidence capture and CSV incident logs.
- Added FastAPI/PostgreSQL enterprise scaffolding with authentication, alert/event APIs, Docker support, and deployment documentation.

## License

MIT License. See [LICENSE](LICENSE).
