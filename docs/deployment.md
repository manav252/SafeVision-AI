# Deployment Guide

## Local Docker Deployment

```bash
cd SafeVision-AI
cp .env.example .env
docker compose up --build
```

Open:

- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## Streamlit Cloud Demo Deployment

For hackathon submission, deploy the Streamlit app from GitHub:

- Main file path: `SafeVision-AI/app.py`
- Python dependencies: `SafeVision-AI/requirements.txt`
- Keep sample videos lightweight and licensed for demo use.

## Backend Deployment Options

The FastAPI backend can be deployed to any Docker-compatible platform such as Render, Railway, Fly.io, Azure App Service, or AWS ECS.

Set:

```text
DATABASE_URL=postgresql+psycopg2://user:password@host:5432/database
SECRET_KEY=<secure-random-secret>
```

## Production Checklist

- Replace default `SECRET_KEY`
- Use HTTPS
- Use managed PostgreSQL
- Add object storage for evidence screenshots
- Configure alert channels such as email, WhatsApp, or SMS
- Add monitoring and log retention
- Use a trained PPE model at `models/ppe_yolov8.pt`
- Validate CCTV/RTSP streams from customer environment

