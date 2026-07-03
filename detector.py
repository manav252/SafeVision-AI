from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / "outputs" / "ultralytics"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    missing_helmet: bool = False
    missing_vest: bool = False
    helmet_estimated: bool = False
    vest_estimated: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)


class SafetyDetector:
    """YOLOv8 detector with PPE-model support and color-heuristic fallback."""

    def __init__(self, models_dir: Path, confidence: float = 0.35) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.confidence = confidence
        self.custom_model_path = self.models_dir / "ppe_yolov8.pt"
        self.fallback_model_path = self.models_dir / "yolov8n.pt"
        self.device = self._select_device()
        self.fallback_mode = not self.custom_model_path.exists()
        self.person_model = YOLO(str(self.fallback_model_path))
        self.ppe_model = None if self.fallback_mode else YOLO(str(self.custom_model_path))

    def _select_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        resized, scale = self._resize_for_inference(frame, width=640)
        person_results = self.person_model.predict(
            source=resized,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        detections = self._parse_yolo_results(person_results[0], scale) if person_results else []
        if self.fallback_mode:
            return self._apply_fallback_ppe(frame, detections)

        ppe_results = self.ppe_model.predict(
            source=resized,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        ppe_detections = self._parse_yolo_results(ppe_results[0], scale) if ppe_results else []
        person_detections = [det for det in detections if det.class_name == "person"]
        return self._attach_custom_ppe_state(person_detections + ppe_detections)

    def _resize_for_inference(self, frame: np.ndarray, width: int) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if w <= width:
            return frame, 1.0
        scale = width / w
        resized = cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _parse_yolo_results(self, result, scale: float) -> list[Detection]:
        names = result.names
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self._normalize_class_name(str(names.get(class_id, class_id)))
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
            bbox = (
                int(x1 / scale),
                int(y1 / scale),
                int(x2 / scale),
                int(y2 / scale),
            )
            detections.append(Detection(class_name=class_name, confidence=conf, bbox=bbox))
        return detections

    def _normalize_class_name(self, class_name: str) -> str:
        normalized = class_name.lower().replace("_", " ").replace("-", " ")
        return " ".join(normalized.split())

    def _attach_custom_ppe_state(self, detections: list[Detection]) -> list[Detection]:
        persons = [d for d in detections if d.class_name == "person"]
        helmets = [d for d in detections if d.class_name in {"helmet", "hardhat", "hard hat"}]
        no_helmets = [d for d in detections if d.class_name in {"no helmet", "no hardhat", "no hard hat"}]
        vests = [d for d in detections if d.class_name in {"safety vest", "vest", "hi-vis vest", "high visibility vest"}]
        no_vests = [d for d in detections if d.class_name in {"no vest", "no safety vest"}]

        for person in persons:
            head_region = self._region_from_bbox(person.bbox, top=0.0, bottom=0.35)
            torso_region = self._region_from_bbox(person.bbox, top=0.25, bottom=0.75)
            person.missing_helmet = self._overlaps_any(head_region, no_helmets) or not self._overlaps_any(head_region, helmets)
            person.missing_vest = self._overlaps_any(torso_region, no_vests) or not self._overlaps_any(torso_region, vests)
        return persons

    def _apply_fallback_ppe(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        persons = [d for d in detections if d.class_name == "person"]
        for person in persons:
            x1, y1, x2, y2 = self._clip_bbox(person.bbox, frame.shape)
            if x2 <= x1 or y2 <= y1:
                continue
            h = y2 - y1
            head = frame[y1 : y1 + int(0.30 * h), x1:x2]
            torso = frame[y1 + int(0.30 * h) : y1 + int(0.70 * h), x1:x2]

            has_helmet = self._has_bright_helmet_cluster(head)
            has_vest = self._has_high_visibility_cluster(torso)
            person.missing_helmet = not has_helmet
            person.missing_vest = not has_vest
            person.helmet_estimated = True
            person.vest_estimated = True
            person.attributes["helmet_signal"] = "yellow_or_white_cluster" if has_helmet else "not_found"
            person.attributes["vest_signal"] = "orange_or_green_cluster" if has_vest else "not_found"
        return persons

    def _has_bright_helmet_cluster(self, region: np.ndarray) -> bool:
        if region.size == 0:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array([18, 70, 120]), np.array([38, 255, 255]))
        white = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([179, 80, 255]))
        mask = cv2.morphologyEx(cv2.bitwise_or(yellow, white), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return self._has_cluster(mask, min_area=max(20, int(region.shape[0] * region.shape[1] * 0.015)))

    def _has_high_visibility_cluster(self, region: np.ndarray) -> bool:
        if region.size == 0:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv, np.array([5, 90, 120]), np.array([24, 255, 255]))
        green = cv2.inRange(hsv, np.array([40, 70, 110]), np.array([85, 255, 255]))
        mask = cv2.morphologyEx(cv2.bitwise_or(orange, green), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return self._has_cluster(mask, min_area=max(35, int(region.shape[0] * region.shape[1] * 0.025)))

    def _has_cluster(self, mask: np.ndarray, min_area: int) -> bool:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(contour) >= min_area for contour in contours)

    def _region_from_bbox(self, bbox: tuple[int, int, int, int], top: float, bottom: float) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        return x1, y1 + int(h * top), x2, y1 + int(h * bottom)

    def _overlaps_any(self, region: tuple[int, int, int, int], detections: list[Detection]) -> bool:
        return any(self._iou(region, det.bbox) > 0.05 for det in detections)

    def _iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union else 0.0

    def _clip_bbox(self, bbox: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        x1, y1, x2, y2 = bbox
        return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)
