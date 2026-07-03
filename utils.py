from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


DEFAULT_ZONE_COLOR = "#ef4444"


def ensure_project_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def build_default_zone(width: int, height: int) -> list[tuple[int, int]]:
    return build_preset_zone(width, height, "bottom_center")


def build_preset_zone(width: int, height: int, placement: str = "bottom_center") -> list[tuple[int, int]]:
    zone_width = int(width * 0.42)
    zone_height = int(height * 0.32)
    if placement == "bottom_left":
        x1 = int(width * 0.04)
    elif placement == "bottom_right":
        x1 = int(width - zone_width - width * 0.04)
    else:
        x1 = int((width - zone_width) / 2)
    y1 = int(height - zone_height - height * 0.06)
    x2 = x1 + zone_width
    y2 = y1 + zone_height
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def extract_polygon_from_canvas(json_data, scale_x: float, scale_y: float) -> list[tuple[int, int]] | None:
    if not json_data or not json_data.get("objects"):
        return None

    objects = json_data.get("objects", [])
    for obj in reversed(objects):
        points = obj.get("path") or obj.get("points")
        if not points:
            continue

        polygon: list[tuple[int, int]] = []
        if obj.get("type") == "path":
            raw_points: list[tuple[float, float]] = []
            for item in points:
                raw_points.extend(_canvas_path_points(item))
            if raw_points:
                polygon = [
                    (int(x * scale_x), int(y * scale_y))
                    for x, y in raw_points
                ]
            polygon = _dedupe_nearby_points(polygon)
            polygon = _close_freehand_polygon(polygon)
        else:
            left = float(obj.get("left", 0))
            top = float(obj.get("top", 0))
            for point in points:
                polygon.append((int((left + point.get("x", 0)) * scale_x), int((top + point.get("y", 0)) * scale_y)))

        if len(polygon) >= 3:
            return polygon
    return None


def _canvas_path_points(path_item) -> list[tuple[float, float]]:
    if len(path_item) < 3:
        return []

    command = path_item[0]
    numeric_values = [value for value in path_item[1:] if isinstance(value, (int, float))]
    if command in {"M", "L"} and len(numeric_values) >= 2:
        return [(float(numeric_values[0]), float(numeric_values[1]))]

    # Freehand brush strokes may use quadratic or cubic curve commands. The final
    # coordinate pair is the point the curve reaches, which is enough for zone hit tests.
    if command in {"Q", "C"} and len(numeric_values) >= 2:
        x, y = numeric_values[-2], numeric_values[-1]
        return [(float(x), float(y))]

    return []


def _dedupe_nearby_points(points: list[tuple[int, int]], min_distance: int = 4) -> list[tuple[int, int]]:
    if not points:
        return []
    deduped = [points[0]]
    for point in points[1:]:
        px, py = deduped[-1]
        if abs(point[0] - px) >= min_distance or abs(point[1] - py) >= min_distance:
            deduped.append(point)
    return deduped


def _close_freehand_polygon(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(points) < 3:
        return points
    if points[0] != points[-1]:
        return points + [points[0]]
    return points


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def person_touches_zone(bbox: tuple[int, int, int, int], polygon: list[tuple[int, int]] | None) -> bool:
    if not polygon or len(polygon) < 3:
        return False

    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    contact_points = [
        (cx, int(y1 + (y2 - y1) * 0.75)),
        (cx, y2),
    ]
    if any(point_in_polygon(point, polygon) for point in contact_points):
        return True

    return bbox_polygon_overlap_ratio(bbox, polygon) >= 0.35


def point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]] | None) -> bool:
    if not polygon or len(polygon) < 3:
        return False
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def bbox_overlaps_polygon(bbox: tuple[int, int, int, int], polygon: list[tuple[int, int]] | None) -> bool:
    if not polygon or len(polygon) < 3:
        return False

    x1, y1, x2, y2 = bbox
    bbox_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), bbox_center(bbox)]
    if any(point_in_polygon(point, polygon) for point in bbox_points):
        return True

    if any(x1 <= px <= x2 and y1 <= py <= y2 for px, py in polygon):
        return True

    bbox_edges = list(zip(bbox_points[:4], bbox_points[1:4] + bbox_points[:1]))
    polygon_edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    return any(_segments_intersect(a, b, c, d) for a, b in bbox_edges for c, d in polygon_edges)


def bbox_polygon_overlap_ratio(bbox: tuple[int, int, int, int], polygon: list[tuple[int, int]] | None) -> float:
    if not polygon or len(polygon) < 3:
        return 0.0

    x1, y1, x2, y2 = bbox
    poly = np.array(polygon, dtype=np.int32)
    min_x = int(min(x1, np.min(poly[:, 0])))
    min_y = int(min(y1, np.min(poly[:, 1])))
    max_x = int(max(x2, np.max(poly[:, 0])))
    max_y = int(max(y2, np.max(poly[:, 1])))

    width = max(1, max_x - min_x + 3)
    height = max(1, max_y - min_y + 3)
    offset = np.array([min_x - 1, min_y - 1], dtype=np.int32)

    polygon_mask = np.zeros((height, width), dtype=np.uint8)
    bbox_mask = np.zeros((height, width), dtype=np.uint8)
    shifted_poly = (poly - offset).reshape(-1, 1, 2)
    cv2.fillPoly(polygon_mask, [shifted_poly], 255)
    cv2.rectangle(
        bbox_mask,
        (int(x1 - offset[0]), int(y1 - offset[1])),
        (int(x2 - offset[0]), int(y2 - offset[1])),
        255,
        -1,
    )

    intersection = int(np.count_nonzero(cv2.bitwise_and(polygon_mask, bbox_mask)))
    smaller_area = max(1, min(int(np.count_nonzero(polygon_mask)), int(np.count_nonzero(bbox_mask))))
    return intersection / smaller_area


def _segments_intersect(a, b, c, d) -> bool:
    def orientation(p, q, r):
        value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if value == 0:
            return 0
        return 1 if value > 0 else 2

    def on_segment(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(a, c, b):
        return True
    if o2 == 0 and on_segment(a, d, b):
        return True
    if o3 == 0 and on_segment(c, a, d):
        return True
    if o4 == 0 and on_segment(c, b, d):
        return True
    return False


def draw_frame_annotations(frame, detections, zone_points, risk_score: int, zone_defs: list[dict] | None = None):
    annotated = frame.copy()
    zones_to_draw = zone_defs or [{"name": "Restricted Zone", "points": zone_points}]
    zone_colors = {
        "Zone A": (0, 0, 255),
        "Zone B": (0, 140, 255),
        "Control Room": (255, 160, 0),
        "Reactor Zone": (168, 85, 247),
        "Restricted Zone": (0, 0, 255),
    }
    for zone_def in zones_to_draw:
        points = zone_def.get("points") if isinstance(zone_def, dict) else None
        if not points or len(points) < 3:
            continue
        zone = np.array(points, dtype=np.int32)
        name = str(zone_def.get("name", "Restricted Zone")) if isinstance(zone_def, dict) else "Restricted Zone"
        color = zone_colors.get(name, (0, 0, 255))
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [zone], color)
        annotated = cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0)
        cv2.polylines(annotated, [zone], True, color, 3)
        label_x = max(8, int(np.min(zone[:, 0]) + 8))
        label_y = max(28, int(np.min(zone[:, 1]) - 8))
        if label_y < 34:
            label_y = min(annotated.shape[0] - 8, int(np.min(zone[:, 1]) + 28))
        label = name.upper() if name != "Restricted Zone" else "RESTRICTED ZONE"
        cv2.putText(annotated, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        labels = ["person"]
        color = (34, 197, 94)

        if detection.missing_helmet:
            labels.append("NO HELMET" + (" est." if detection.helmet_estimated else ""))
            color = (0, 215, 255)
        if detection.missing_vest:
            labels.append("NO VEST" + (" est." if detection.vest_estimated else ""))
            color = (0, 165, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = " | ".join(labels)
        _draw_label(annotated, label, x1, max(22, y1 - 8), color)

    risk_color = (34, 197, 94) if risk_score < 35 else (0, 215, 255) if risk_score < 70 else (0, 0, 255)
    cv2.rectangle(annotated, (12, 12), (220, 58), (17, 24, 39), -1)
    cv2.putText(annotated, f"Risk: {risk_score}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, risk_color, 2)
    return annotated


def _draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_top = max(0, y - th - baseline - 6)
    x_right = min(frame.shape[1] - 1, x + tw + 10)
    cv2.rectangle(frame, (x, y_top), (x_right, y + baseline), color, -1)
    cv2.putText(frame, text, (x + 5, y - 5), font, scale, (17, 24, 39), thickness)


def save_evidence_frame(frame, evidence_dir: Path, frame_index: int, risk_score: int) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = evidence_dir / f"evidence_frame_{frame_index}_risk_{risk_score}_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def write_violation_log_csv(rows: list[dict], logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"violation_log_{timestamp}.csv"
    columns = ["frame", "timestamp", "violation_type", "severity", "risk_points", "estimated", "message"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path
