# Docker Guide

1. Copy the environment template:

```bash
cp .env.example .env
```

2. Edit `.env` and set a unique `JWT_SECRET_KEY` and database password.

3. Start the stack:

```bash
docker compose up --build
```

Services:

- FastAPI: `http://localhost:8000`
- Swagger/OpenAPI: `http://localhost:8000/docs`
- Streamlit dashboard: `http://localhost:8501`
- PostgreSQL: `localhost:5432`

The FastAPI service installs `requirements-backend.txt`. The Streamlit service installs `requirements.txt` so the existing demo remains lightweight.
