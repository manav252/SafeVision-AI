from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import Base, engine
from .logging_config import configure_logging
from .routers import advisor, alerts, auth, cameras, events, health, heatmap

configure_logging()
settings = get_settings()

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SafeVision AI API",
    description="Industrial safety intelligence APIs for CCTV, PPE, restricted-zone, gas, permit, and response workflows.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(auth.router, prefix=f"{settings.api_prefix}/auth", tags=["Authentication"])
app.include_router(cameras.router, prefix=f"{settings.api_prefix}/cameras", tags=["Cameras"])
app.include_router(events.router, prefix=f"{settings.api_prefix}/events", tags=["Safety Events"])
app.include_router(alerts.router, prefix=f"{settings.api_prefix}/alerts", tags=["Alerts"])
app.include_router(heatmap.router, prefix=f"{settings.api_prefix}/heatmap", tags=["Heatmap"])
app.include_router(advisor.router, prefix=f"{settings.api_prefix}/advisor", tags=["AI Safety Advisor"])

