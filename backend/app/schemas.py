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


class PpeStatus(BaseModel):
    helmet: bool = True
    vest: bool = True
    gloves: bool | None = None
    boots: bool | None = None


class GasReadings(BaseModel):
    methane_lel: float = Field(default=0, ge=0)
    co_ppm: float = Field(default=0, ge=0)
    h2s_ppm: float = Field(default=0, ge=0)
    oxygen_percent: float = Field(default=20.9, ge=0, le=100)


class ZoneStatus(BaseModel):
    zone_name: str = "Plant"
    restricted_zone_breach: bool = False


class DetectionCreate(BaseModel):
    camera_id: UUID | None = None
    worker_id: str | None = None
    detection_type: str = "person"
    confidence_score: float = Field(ge=0, le=1)
    ppe_status: PpeStatus = Field(default_factory=PpeStatus)
    gas_readings: GasReadings = Field(default_factory=GasReadings)
    zone_status: ZoneStatus = Field(default_factory=ZoneStatus)
    metadata: dict | None = None


class DetectionRead(BaseModel):
    event: SafetyEventRead
    calculated_risk_score: int
    alert_created: bool
    risk_factors: list[str]


class EventsSummary(BaseModel):
    total_events: int
    by_severity: dict[str, int]
    by_type: dict[str, int]


class AlertsSummary(BaseModel):
    total_alerts: int
    active_alerts: int
    by_status: dict[str, int]
    by_severity: dict[str, int]


class SafetyReport(BaseModel):
    generated_at: datetime
    events: EventsSummary
    alerts: AlertsSummary
    recent_events: list[SafetyEventRead]
    open_alerts: list[AlertRead]


class DashboardSummary(BaseModel):
    total_events: int
    active_alerts: int
    risk_distribution: dict[str, int]
    recent_incidents: list[SafetyEventRead]
    heatmap_summary: dict
