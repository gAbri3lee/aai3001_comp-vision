"""
Safety decision & response engine — rule-based, NOT a trained model by
design (see CLAUDE.md pipeline stage 3). Fuses detection + depth, ranks
hazards by severity/proximity/position, and applies a cooldown so the same
hazard isn't re-announced every frame.
"""
import time
from dataclasses import dataclass

from depth import box_proximity

# Rough severity weights: moving/vehicular hazards outrank static furniture.
# Placeholder tuned for COCO classes; revisit once real hazard classes exist.
SEVERITY = {
    "car": 1.0, "bus": 1.0, "truck": 1.0, "motorcycle": 0.9, "bicycle": 0.8,
    "person": 0.6, "dog": 0.5,
    "traffic light": 0.4, "stop sign": 0.4, "fire hydrant": 0.3,
    "bench": 0.2, "chair": 0.2, "couch": 0.2, "potted plant": 0.15,
    "backpack": 0.15, "suitcase": 0.15,
}
DEFAULT_SEVERITY = 0.3

COOLDOWN_SECONDS = 4.0
PROXIMITY_JUMP_TO_REALERT = 0.15


@dataclass
class Hazard:
    label: str
    proximity: float  # 0-1, higher = closer
    position: str      # "left" / "center" / "right"
    priority: float
    xyxy: tuple


def _position_bucket(x_center_norm: float) -> str:
    if x_center_norm < 1 / 3:
        return "left"
    if x_center_norm > 2 / 3:
        return "right"
    return "center"


def build_hazards(results, depth_map, frame_width: int) -> list[Hazard]:
    hazards = []
    names = results.names
    for box in results.boxes:
        # Lowercased so this works whether the loaded model's class names
        # are COCO-style lowercase (placeholder yolov8n.pt) or Title Case
        # (the Open Images-trained hazard checkpoint) — keeps SEVERITY keys
        # stable across a model swap instead of needing to duplicate them.
        label = names[int(box.cls)].lower()
        xyxy = box.xyxy[0].tolist()
        proximity = box_proximity(depth_map, xyxy)
        x_center_norm = ((xyxy[0] + xyxy[2]) / 2) / frame_width
        position = _position_bucket(x_center_norm)
        severity = SEVERITY.get(label, DEFAULT_SEVERITY)
        priority = severity * (0.5 + 0.5 * proximity)
        hazards.append(Hazard(label, proximity, position, priority, tuple(xyxy)))
    return sorted(hazards, key=lambda h: h.priority, reverse=True)


class AlertCooldown:
    """Tracks the last time each (label, position) hazard was announced so
    the TTS stage doesn't repeat itself every frame — unless the hazard has
    gotten meaningfully closer, in which case it re-alerts early."""

    def __init__(self, cooldown_seconds: float = COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self._last: dict[tuple[str, str], tuple[float, float]] = {}

    def should_announce(self, hazard: Hazard) -> bool:
        key = (hazard.label, hazard.position)
        now = time.time()
        last = self._last.get(key)
        if last is None:
            self._last[key] = (now, hazard.proximity)
            return True
        last_time, last_proximity = last
        got_closer = hazard.proximity - last_proximity > PROXIMITY_JUMP_TO_REALERT
        if now - last_time > self.cooldown_seconds or got_closer:
            self._last[key] = (now, hazard.proximity)
            return True
        return False
