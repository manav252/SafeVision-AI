import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    safety_officer = "safety_officer"
    operator = "operator"
    viewer = "viewer"


class AlertSeverity(str, enum.Enum):
    low = "LOW"
    medium = "MEDIUM"
    high = "HIGH"
    critical = "CRITICAL"


class AlertStatus(str, enum.Enum):
    open = "OPEN"
    acknowledged = "ACKNOWLEDGED"
    resolved = "RESOLVED"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.viewer)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120), index=True)
    stream_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    zone_name: Mapped[str] = mapped_column(String(120), default="Pending Setup")
    status: Mapped[str] = mapped_column(String(40), default="Pending Setup")
    restricted_zone: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    events: Mapped[list["SafetyEvent"]] = relationship(back_populates="camera")


class SafetyEvent(Base):
    __tablename__ = "safety_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("cameras.id"), nullable=True)
    zone_name: Mapped[str] = mapped_column(String(120), default="Plant")
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    severity: Mapped[AlertSeverity] = mapped_column(Enum(AlertSeverity), default=AlertSeverity.low)
    message: Mapped[str] = mapped_column(Text)
    risk_score: Mapped[int] = mapped_column(Integer, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    evidence_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    camera: Mapped[Camera | None] = relationship(back_populates="events")
    alert: Mapped["Alert | None"] = relationship(back_populates="event")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("safety_events.id"))
    title: Mapped[str] = mapped_column(String(160))
    severity: Mapped[AlertSeverity] = mapped_column(Enum(AlertSeverity))
    status: Mapped[AlertStatus] = mapped_column(Enum(AlertStatus), default=AlertStatus.open)
    assigned_to: Mapped[str | None] = mapped_column(String(255), nullable=True)
    response_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    event: Mapped[SafetyEvent] = relationship(back_populates="alert")


class PlantSignal(Base):
    __tablename__ = "plant_signals"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    zone_name: Mapped[str] = mapped_column(String(120), index=True)
    methane_lel: Mapped[float] = mapped_column(Float, default=0)
    co_ppm: Mapped[float] = mapped_column(Float, default=0)
    h2s_ppm: Mapped[float] = mapped_column(Float, default=0)
    oxygen_percent: Mapped[float] = mapped_column(Float, default=20.9)
    permit_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    equipment_status: Mapped[str | None] = mapped_column(String(160), nullable=True)
    shift_status: Mapped[str | None] = mapped_column(String(160), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
