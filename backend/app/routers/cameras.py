from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user, require_roles
from ..models import Camera, User, UserRole
from ..schemas import CameraCreate, CameraRead

router = APIRouter()


@router.post("/", response_model=CameraRead)
def create_camera(
    payload: CameraCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(UserRole.admin, UserRole.safety_officer)),
) -> Camera:
    camera = Camera(**payload.model_dump(), status="Configured" if payload.restricted_zone else "Pending Setup")
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


@router.get("/", response_model=list[CameraRead])
def list_cameras(db: Session = Depends(get_db), _: User = Depends(current_user)) -> list[Camera]:
    return db.query(Camera).order_by(Camera.created_at.desc()).all()


@router.patch("/{camera_id}/zone", response_model=CameraRead)
def update_camera_zone(
    camera_id: UUID,
    payload: CameraCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(UserRole.admin, UserRole.safety_officer)),
) -> Camera:
    camera = db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    camera.zone_name = payload.zone_name
    camera.restricted_zone = payload.restricted_zone
    camera.status = "Configured" if payload.restricted_zone else "Pending Setup"
    db.commit()
    db.refresh(camera)
    return camera

