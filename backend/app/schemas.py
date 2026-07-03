from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from .models import AlertSeverity, AlertStatus, UserRole


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str = Field(min_length=8)
    role: UserRole = UserRole.viewer


class UserRead(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool

    class Config:
        from_attributes = True


class CameraCreate(BaseModel):
    name: str
    stream_url: str | None = None
    zone_name: str = "Pending Setup"
    restricted_zone: dict | None = None


class CameraRead(CameraCreate):
    id: UUID
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class SafetyEventCreate(BaseModel):
    camera_id: UUID | None = None
    zone_name: str = "Plant"
    event_type: str
    message: str
    worker_id: str | None = None
    evidence_uri: str | None = None
    metadata_json: dict | None = None


class SafetyEventRead(SafetyEventCreate):
    id: UUID
    severity: AlertSeverity
    risk_score: int
    created_at: datetime

    class Config:
        from_attributes = True


class AlertRead(BaseModel):
    id: UUID
    event_id: UUID
    title: str
    severity: AlertSeverity
    status: AlertStatus
    assigned_to: str | None = None
    response_notes: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class PlantSignalCreate(BaseModel):
    zone_name: str
    methane_lel: float = 0
    co_ppm: float = 0
    h2s_ppm: float = 0
    oxygen_percent: float = 20.9
    permit_type: str | None = None
    equipment_status: str | None = None
    shift_status: str | None = None


class RiskSummary(BaseModel):
    risk_score: int
    risk_level: str
    factors: list[str]
    recommendation: str

