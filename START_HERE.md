# SafeVision AI - Start Here

This folder contains the complete SafeVision AI demo:

- `app.py` - Streamlit landing page and live dashboard in one app
- `assets/SafeVision_AI_Architecture.png` - architecture diagram used by the landing page and dashboard
- `sample_videos/` - bundled CCTV demo footage
- `models/` - YOLO model files

Live demo:

https://safevision-ai-manav25.streamlit.app

Run the demo:

```bash
cd SafeVision-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the landing page, then click **Launch Live Dashboard** to enter the SafeVision operations dashboard.
