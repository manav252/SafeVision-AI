# SafeVision AI

SafeVision AI is a portfolio-ready web demo for an AI-powered industrial safety intelligence platform. It shows how CCTV analytics can be fused with plant context such as gas readings, work permits, equipment status, shift notes, restricted zones and checklist completion to help safety teams detect compound risk earlier.

## Project Idea

Factories often have multiple safety systems, but they usually operate separately:

- CCTV detects people but does not understand gas or permit context.
- Gas sensors trigger threshold alarms but do not know worker location.
- Work permits define activity but may not react to live process conditions.
- Compliance checklists record safety controls but are not connected to live CCTV events.

SafeVision AI presents a unified command center where these signals are combined into real-time risk intelligence.

## Features

- Landing page for the SafeVision AI product story
- Demo dashboard with industrial command-center UI
- Plant Camera Manager with upload and sample feed selection
- Restricted zone drawing mock interface
- Plant signal controls for PPE, gas, permit, equipment and checklist status
- Dynamic risk engine with Low / Medium / High / Critical severity
- Live detection mock panel
- Recent Safety Events
- Explain This Alert card with reasoning, contributing factors and recommended actions
- AI Safety Advisor
- Risk Heatmap
- Architecture modal
- Responsive SaaS-style design

## Demo Flow

1. Open the landing page.
2. Click **View Demo Dashboard**.
3. Select a sample CCTV scenario or upload a video name.
4. Toggle plant signal inputs such as PPE violation, restricted-zone entry, gas level and permit status.
5. Click **Start Monitoring**.
6. Show how the risk score, events, explanation card, AI Safety Advisor and heatmap update together.
7. Open the **Architecture** modal to explain the system pipeline.

## Architecture

```text
Multi-camera CCTV feeds
        ↓
Vision Processing
        ↓
Computer Vision Engine
        ↓
Safety Fusion Engine
        ↓
Safety Rule Engine
        ↓
Risk Engine
        ↓
AI Safety Advisor
        ↓
Dashboard + Heatmap + Reports
```

Core principle:

```text
Vision Intelligence + Operational Context + Safety Rules = Real-Time Risk Intelligence
```

## Screenshots

Add screenshots in the `screenshots/` folder before submitting:

- `screenshots/landing-page.png`
- `screenshots/demo-dashboard.png`
- `screenshots/risk-heatmap.png`
- `screenshots/architecture-modal.png`

## Tech Stack

- React
- Vite
- Framer Motion
- Lucide React icons
- CSS responsive design
- Mock data for portfolio/demo use

The repository also contains a production-oriented FastAPI/PostgreSQL backend scaffold for future expansion.

## Run Locally

```bash
npm install
npm run dev
```

Open the local URL printed by Vite.

## Build

```bash
npm run build
```

## Deploy on Vercel

1. Push this repository to GitHub.
2. Import the repo in Vercel.
3. Use these settings:
   - Framework: Vite
   - Build command: `npm run build`
   - Output directory: `dist`
4. Deploy.

## Resume Bullets

- Built SafeVision AI, a React-based industrial safety intelligence demo combining CCTV analytics, gas readings, permit context, equipment status and compliance checklist signals.
- Designed a rule-based risk engine that classifies plant safety states into Low, Medium, High and Critical risk levels.
- Created AI-style alert explanations and recommended actions to improve incident response transparency.
- Developed a responsive SaaS dashboard with camera manager, detection panel, event feed, risk heatmap and architecture modal.
- Prepared the project for Vercel deployment with Vite build tooling and production-ready documentation.

## Disclaimer

This is a hackathon and portfolio demo. CCTV, IoT and permit integrations are mocked for presentation, while the architecture is structured for future real integrations.
