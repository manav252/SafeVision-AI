from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class SyncResult:
    attempted: int = 0
    synced: int = 0
    error: str | None = None


class SafeVisionApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        email: str | None = None,
        password: str | None = None,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("SAFEVISION_API_URL") or "http://localhost:8000").rstrip("/")
        self.email = email or os.getenv("SAFEVISION_API_EMAIL") or "streamlit@safevision.ai"
        self.password = password or os.getenv("SAFEVISION_API_PASSWORD")
        self.timeout_seconds = timeout_seconds
        self._token: str | None = None

    def enabled(self) -> bool:
        return os.getenv("SAFEVISION_BACKEND_SYNC", "true").lower() in {"1", "true", "yes", "on"}

    def sync_events(self, rows: list[dict], gas_context: dict | None = None) -> SyncResult:
        if not self.enabled() or not rows:
            return SyncResult()

        token = self._ensure_token()
        headers = {"Authorization": f"Bearer {token}"}
        result = SyncResult(attempted=len(rows))
        for row in rows:
            response = requests.post(
                f"{self.base_url}/api/v1/detection/",
                json=build_detection_payload(row, gas_context),
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            result.synced += 1
        return result

    def _ensure_token(self) -> str:
        if not self.password:
            raise RuntimeError("SAFEVISION_API_PASSWORD is required when backend sync is enabled")
        if self._token:
            return self._token
        self._register_service_user()
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            data={"username": self.email, "password": self.password},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        self._token = response.json()["access_token"]
        return self._token

    def _register_service_user(self) -> None:
        response = requests.post(
            f"{self.base_url}/api/v1/auth/register",
            json={
                "email": self.email,
                "full_name": "Streamlit Sync User",
                "password": self.password,
                "role": "operator",
            },
            timeout=self.timeout_seconds,
        )
        if response.status_code not in {200, 409}:
            response.raise_for_status()


def build_detection_payload(row: dict[str, Any], gas_context: dict | None = None) -> dict[str, Any]:
    violation_type = str(row.get("violation_type") or "streamlit_event")
    zone_name = str(row.get("zone_name") or "Plant")
    readings = (gas_context or {}).get("readings", {})

    return {
        "worker_id": row.get("worker_id"),
        "detection_type": violation_type,
        "confidence_score": _confidence_score(row),
        "ppe_status": {
            "helmet": violation_type != "no_helmet",
            "vest": violation_type != "no_vest",
        },
        "gas_readings": {
            "methane_lel": float(readings.get("methane_lel") or 0),
            "co_ppm": float(readings.get("co_ppm") or 0),
            "h2s_ppm": float(readings.get("h2s_ppm") or 0),
            "oxygen_percent": float(readings.get("oxygen_pct") or readings.get("oxygen_percent") or 20.9),
        },
        "zone_status": {
            "zone_name": zone_name,
            "restricted_zone_breach": violation_type in {"restricted_zone_entry", "restricted_zone_breach"},
        },
        "metadata": {
            "source": "streamlit",
            "frame": row.get("frame"),
            "timestamp": row.get("timestamp"),
            "severity": row.get("severity"),
            "risk_points": int(row.get("risk_points") or 0),
            "estimated": bool(row.get("estimated", False)),
            "message": row.get("message"),
        },
    }


def _confidence_score(row: dict[str, Any]) -> float:
    raw_value = row.get("confidence") or row.get("confidence_score")
    if raw_value is not None:
        try:
            return max(0.0, min(1.0, float(raw_value)))
        except (TypeError, ValueError):
            pass
    return 0.55 if row.get("estimated") else 0.75
