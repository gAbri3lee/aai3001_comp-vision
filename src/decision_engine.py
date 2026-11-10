"""
Safety decision & response engine — rule-based, NOT a trained model by
design (see CLAUDE.md pipeline stage 3). Fuses detection + depth, ranks
hazards by severity/proximity/position, and applies a cooldown so the same
hazard isn't re-announced every frame.
"""
import time
from dataclasses import dataclass

from config import load_config

# Application heuristics; these are not learned model scores.
SEVERITY = load_config()["severity"]

COOLDOWN_SECONDS = 4.0
PROXIMITY_JUMP_TO_REALERT = 0.15


@dataclass
class Hazard:
    label: str
    proximity: float | None  # Relative 0-1, or unavailable without depth
    position: str      # "left" / "center" / "right"
    priority: float
    xyxy: tuple


def _position_bucket(x_center_norm: float) -> str:
    if x_center_norm < 1 / 3:
        return "left"
    if x_center_norm > 2 / 3:
        return "right"
    return "center"


def build_hazards(detections, depth_map, frame_width: int, severity=None) -> list[Hazard]:
    severity = SEVERITY if severity is None else severity
    hazards = []
    for detection in detections:
        label, xyxy = detection.label, detection.box_xyxy
        proximity = None
        if depth_map is not None:
            from depth import box_proximity
            proximity = box_proximity(depth_map, xyxy)
        position = _position_bucket(((xyxy[0] + xyxy[2]) / 2) / frame_width)
        priority = severity[label] * (0.5 + 0.5 * (proximity if proximity is not None else 0.0))
        hazards.append(Hazard(label, proximity, position, priority, tuple(xyxy)))
    return sorted(hazards, key=lambda h: h.priority, reverse=True)


def phrase_for(hazard) -> str:
    position = "ahead" if hazard.position == "center" else f"on your {hazard.position}"
    if hazard.proximity is None:
        return f"Note, {hazard.label} {position}; distance unavailable"
    urgency = "Warning" if hazard.proximity > 0.7 else "Caution" if hazard.proximity > 0.4 else "Note"
    return f"{urgency}, {hazard.label} {position}"


class AlertCooldown:
    """Tracks the last time each (label, position) hazard was announced so
    the TTS stage doesn't repeat itself every frame — unless the hazard has
    gotten meaningfully closer, in which case it re-alerts early."""

    def __init__(self, cooldown_seconds: float = COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self._last: dict[tuple[str, str], tuple[float, float]] = {}

    def should_announce(self, hazard: Hazard) -> bool:
        key = (hazard.label, hazard.position)
        now = time.monotonic()
        last = self._last.get(key)
        if last is None:
            self._last[key] = (now, hazard.proximity)
            return True
        last_time, last_proximity = last
        got_closer = (hazard.proximity is not None and last_proximity is not None
                      and hazard.proximity - last_proximity > PROXIMITY_JUMP_TO_REALERT)
        if now - last_time > self.cooldown_seconds or got_closer:
            self._last[key] = (now, hazard.proximity)
            return True
        return False
