# SafeVision AI Quickstart

## Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-backend.txt
cp .env.example .env
uvicorn backend.app.main:app --reload
```

Open `http://localhost:8000/docs`.

## Streamlit Demo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Tests

```bash
pip install -r requirements-dev.txt
JWT_SECRET_KEY=test-secret-key-for-safevision-ai-32-chars pytest
```
