from __future__ import annotations

from dataclasses import dataclass

from detector import Detection
from utils import person_touches_zone


@dataclass
class RiskResult:
    score: int
    level: str
    violations: list[dict]


class RiskEngine:
    RESTRICTED_ZONE_POINTS = 40
    NO_HELMET_POINTS = 30
    NO_VEST_POINTS = 25
    GAS_ACCUMULATION_POINTS = 35
    COMPOUND_GAS_WORK_POINTS = 45
    FIRE_OVERRIDE_POINTS = 50
    EQUIPMENT_MAINTENANCE_POINTS = 20
    SHIFT_CHANGEOVER_POINTS = 15
    AUDIT_DEVIATION_POINTS = 20
    WORKER_REGISTRY = ["Amit Sharma", "Priya Nair", "Rahul Verma", "Neha Patil", "Karan Mehta"]

    def evaluate_frame(
        self,
        frame_index: int,
        timestamp_seconds: float,
        detections: list[Detection],
        zone_points: list[tuple[int, int]],
        fallback_mode: bool,
        gas_context: dict | None = None,
    ) -> RiskResult:
        score = 0
        violations: list[dict] = []

        gas_result = self.evaluate_gas_context(frame_index, timestamp_seconds, gas_context)
        score += gas_result.score
        violations.extend(gas_result.violations)

        for person_id, detection in enumerate(detections, start=1):
            worker_name = self._worker_name(person_id)
            if person_touches_zone(detection.bbox, zone_points):
                score += self.RESTRICTED_ZONE_POINTS
                violations.append(
                    self._row(
                        frame_index,
                        timestamp_seconds,
                        "restricted_zone_breach",
                        "HIGH",
                        self.RESTRICTED_ZONE_POINTS,
                        False,
                        f"ZONE ENTRY ALERT (+{self.RESTRICTED_ZONE_POINTS}) | Worker {worker_name} entered restricted zone. "
                        "Action: notify area supervisor and verify worker exit.",
                    )
                )

            if detection.missing_helmet:
                score += self.NO_HELMET_POINTS
                violations.append(
                    self._row(
                        frame_index,
                        timestamp_seconds,
                        "no_helmet",
                        "MEDIUM",
                        self.NO_HELMET_POINTS,
                        detection.helmet_estimated or fallback_mode,
                        f"ADMIN WARNING SENT | Worker {worker_name} helmet missing or removed.",
                    )
                )

            if detection.missing_vest:
                score += self.NO_VEST_POINTS
                violations.append(
                    self._row(
                        frame_index,
                        timestamp_seconds,
                        "no_vest",
                        "MEDIUM",
                        self.NO_VEST_POINTS,
                        detection.vest_estimated or fallback_mode,
                        f"ADMIN WARNING SENT | Worker {worker_name} safety vest missing.",
                    )
                )

        capped_score = min(100, score)
        return RiskResult(score=capped_score, level=self._level(capped_score), violations=violations)

    def evaluate_gas_context(
        self,
        frame_index: int,
        timestamp_seconds: float,
        gas_context: dict | None,
    ) -> RiskResult:
        if not gas_context or not gas_context.get("enabled", False):
            return RiskResult(score=0, level="LOW", violations=[])

        readings = gas_context.get("readings", {})
        active_work = bool(gas_context.get("maintenance_active")) or gas_context.get("permit_type") != "None"
        fire_detected = bool(gas_context.get("fire_detected", False))
        equipment_status = gas_context.get("equipment_status", "Normal")
        shift_phase = gas_context.get("shift_phase", "Stable operations")
        audit_status = gas_context.get("audit_status", "Compliant")
        elevated_reasons = self._gas_elevated_reasons(readings)
        critical_reasons = self._gas_critical_reasons(readings)

        score = 0
        violations: list[dict] = []

        if elevated_reasons:
            severity = "HIGH" if critical_reasons else "MEDIUM"
            score += self.GAS_ACCUMULATION_POINTS
            violations.append(
                self._row(
                    frame_index,
                    timestamp_seconds,
                    "hazardous_gas_accumulation",
                    severity,
                    self.GAS_ACCUMULATION_POINTS,
                    True,
                    f"GAS ALERT (+{self.GAS_ACCUMULATION_POINTS}) | Abnormal atmosphere detected: "
                    + ", ".join(elevated_reasons)
                    + ". Action: verify sensor reading and increase ventilation.",
                )
            )

        if elevated_reasons and active_work:
            score += self.COMPOUND_GAS_WORK_POINTS
            permit = gas_context.get("permit_type", "active work")
            violations.append(
                self._row(
                    frame_index,
                    timestamp_seconds,
                    "compound_gas_work_permit_risk",
                    "HIGH",
                    self.COMPOUND_GAS_WORK_POINTS,
                    True,
                    f"PERMIT CONFLICT (+{self.COMPOUND_GAS_WORK_POINTS}) | {permit} is active while gas is elevated. "
                    "Action: pause permit approval and notify safety supervisor.",
                )
            )

        if elevated_reasons and equipment_status != "Normal":
            score += self.EQUIPMENT_MAINTENANCE_POINTS
            violations.append(
                self._row(
                    frame_index,
                    timestamp_seconds,
                    "equipment_condition_overlap",
                    "HIGH",
                    self.EQUIPMENT_MAINTENANCE_POINTS,
                    True,
                    f"EQUIPMENT RISK (+{self.EQUIPMENT_MAINTENANCE_POINTS}) | {equipment_status} during elevated gas. "
                    "Action: confirm isolation/lockout before work continues.",
                )
            )

        if elevated_reasons and shift_phase != "Stable operations":
            score += self.SHIFT_CHANGEOVER_POINTS
            violations.append(
                self._row(
                    frame_index,
                    timestamp_seconds,
                    "shift_changeover_risk",
                    "MEDIUM",
                    self.SHIFT_CHANGEOVER_POINTS,
                    True,
                    f"SHIFT HANDOVER (+{self.SHIFT_CHANGEOVER_POINTS}) | {shift_phase} during abnormal gas. "
                    "Action: incoming supervisor must acknowledge the active hazard.",
                )
            )

        if audit_status != "Compliant" and (elevated_reasons or active_work):
            score += self.AUDIT_DEVIATION_POINTS
            violations.append(
                self._row(
                    frame_index,
                    timestamp_seconds,
                    "compliance_audit_deviation",
                    "MEDIUM",
                    self.AUDIT_DEVIATION_POINTS,
                    True,
                    f"COMPLIANCE GAP (+{self.AUDIT_DEVIATION_POINTS}) | {audit_status} while active work/hazard exists. "
                    "Action: close checklist item before incident closure.",
                )
            )

        if fire_detected:
            score += self.FIRE_OVERRIDE_POINTS
            violations.append(
                self._row(
                    frame_index,
                    timestamp_seconds,
                    "fire_biometric_override",
                    "HIGH",
                    self.FIRE_OVERRIDE_POINTS,
                    True,
                    "EMERGENCY OVERRIDE SENT | Fire detected. Biometric locks disabled for evacuation and administrator notified.",
                )
            )

        capped_score = min(100, score)
        return RiskResult(score=capped_score, level=self._level(capped_score), violations=violations)

    def _worker_name(self, person_id: int) -> str:
        return self.WORKER_REGISTRY[(person_id - 1) % len(self.WORKER_REGISTRY)]

    def _gas_elevated_reasons(self, readings: dict) -> list[str]:
        reasons = []
        if readings.get("methane_lel", 0) >= 10:
            reasons.append(f"CH4 {readings.get('methane_lel')}% LEL")
        if readings.get("co_ppm", 0) >= 35:
            reasons.append(f"CO {readings.get('co_ppm')} ppm")
        if readings.get("h2s_ppm", 0) >= 10:
            reasons.append(f"H2S {readings.get('h2s_ppm')} ppm")
        oxygen = readings.get("oxygen_pct", 20.9)
        if oxygen < 19.5 or oxygen > 23.5:
            reasons.append(f"O2 {oxygen}%")
        return reasons

    def _gas_critical_reasons(self, readings: dict) -> list[str]:
        reasons = []
        if readings.get("methane_lel", 0) >= 20:
            reasons.append("critical methane")
        if readings.get("co_ppm", 0) >= 100:
            reasons.append("critical CO")
        if readings.get("h2s_ppm", 0) >= 20:
            reasons.append("critical H2S")
        if readings.get("oxygen_pct", 20.9) < 19.0:
            reasons.append("oxygen deficient")
        return reasons

    def _level(self, score: int) -> str:
        if score >= 70:
            return "HIGH"
        if score >= 35:
            return "MEDIUM"
        return "LOW"

    def _row(
        self,
        frame_index: int,
        timestamp_seconds: float,
        violation_type: str,
        severity: str,
        risk_points: int,
        estimated: bool,
        message: str,
    ) -> dict:
        suffix = " (estimated)" if estimated else ""
        return {
            "frame": frame_index,
            "timestamp": f"{timestamp_seconds:.2f}s",
            "violation_type": violation_type,
            "severity": severity,
            "risk_points": risk_points,
            "estimated": estimated,
            "message": f"{message}{suffix}",
        }
