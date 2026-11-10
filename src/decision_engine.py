"""
Safety decision and response engine. It is rule based by design, not a trained
model. It turns detections into hazards, applies the alert gate (size, path and
confidence rules), ranks what is left and applies a cooldown so the same hazard
is not announced every frame.

The same functions process local and cloud detections, so both backends follow
identical alert rules. Detections are boxes in original-frame pixels.
"""
import time
from dataclasses import dataclass

from config import load_config

# Application heuristics; these are not learned model scores.
_CONFIG = load_config()
SEVERITY = _CONFIG["severity"]
ALERT_GATE = _CONFIG["alert_gate"]

COOLDOWN_SECONDS = 4.0
PROXIMITY_JUMP_TO_REALERT = 0.15
OBJECT_WEIGHT = 0.3

# Band names from far to near, and the wording each announced band uses.
BANDS = ("far", "mid", "mid-close", "near")
BAND_WORD = {"near": "Warning", "mid-close": "Caution", "mid": "Note"}


@dataclass
class Hazard:
    label: str
    proximity: float | None  # Relative 0-1 (depth, or box size when the gate is on); None if unavailable
    position: str      # "left" / "center" / "right"
    priority: float
    xyxy: tuple
    size: float | None = None        # Box height divided by frame height
    band: str | None = None          # far / mid / mid-close / near; None when the gate is disabled
    source: str = "trained"          # "trained" (five-class model) or "general" (COCO model)
    confidence: float | None = None
    class_id: int | None = None
    gated: bool = True               # True: passed the alert gate and may be announced
    reason: str | None = None        # Why it was ignored: far / side / low_confidence

    @property
    def key(self):
        return (self.label, self.position, self.band)

    def to_dict(self):
        return {"class_id": self.class_id, "class": self.label,
                "confidence": self.confidence, "box_xyxy": list(self.xyxy),
                "position": self.position, "size": self.size, "band": self.band,
                "source": self.source, "gated": self.gated, "reason": self.reason,
                "proximity": self.proximity, "priority": self.priority,
                "phrase": phrase_for(self) if self.gated else None}


def _position_bucket(x_center_norm: float, centre=(1 / 3, 2 / 3)) -> str:
    if x_center_norm < centre[0]:
        return "left"
    if x_center_norm > centre[1]:
        return "right"
    return "center"


def _source(detection):
    return "general" if detection.class_id < 0 else "trained"


def build_hazards(detections, depth_map, frame_width: int, severity=None) -> list[Hazard]:
    """Ungated hazards, used when the alert gate is disabled."""
    severity = SEVERITY if severity is None else severity
    hazards = []
    for detection in detections:
        label, xyxy = detection.label, detection.box_xyxy
        proximity = None
        if depth_map is not None:
            from depth import box_proximity
            proximity = box_proximity(depth_map, xyxy)
        position = _position_bucket(((xyxy[0] + xyxy[2]) / 2) / frame_width)
        weight = OBJECT_WEIGHT if label == 'Object' else severity[label]
        priority = weight * (0.5 + 0.5 * (proximity if proximity is not None else 0.0))
        hazards.append(Hazard(label, proximity, position, priority, tuple(xyxy),
                              source=_source(detection), confidence=detection.confidence,
                              class_id=detection.class_id))
    return sorted(hazards, key=lambda h: h.priority, reverse=True)


def _band(size, near_size, depth_proximity):
    # Depth and box size are never mixed: with depth on, urgency follows depth only.
    if depth_proximity is not None:
        return "near" if depth_proximity > 0.7 else "mid-close" if depth_proximity > 0.4 else "mid"
    ratio = size / near_size
    return "near" if ratio >= 1.0 else "mid-close" if ratio >= 0.6 else "mid"


def assess(detections, frame_size, gate=None, severity=None, depth_map=None) -> list[Hazard]:
    """Apply the alert gate to every detection and rank the result.

    Pure function: no I/O, no clock. Returns one Hazard per detection. Hazards
    that passed the gate (`gated` True) come first, trained classes before generic
    Objects, then by priority. Ignored hazards keep a `reason`.
    """
    gate = ALERT_GATE if gate is None else gate
    severity = SEVERITY if severity is None else severity
    width, height = frame_size
    if not gate["enabled"]:
        return build_hazards(detections, depth_map, width, severity)
    centre = tuple(gate["centre_third"])
    hazards = []
    for detection in detections:
        label, xyxy = detection.label, tuple(detection.box_xyxy)
        source = _source(detection)
        size = (xyxy[3] - xyxy[1]) / height
        position = _position_bucket(((xyxy[0] + xyxy[2]) / 2) / width, centre)
        min_size, near_size = gate["min_size"][label], gate["near_size"][label]
        depth_proximity = None
        if depth_map is not None:
            from depth import box_proximity
            depth_proximity = box_proximity(depth_map, xyxy)
        proximity = depth_proximity if depth_proximity is not None else min(1.0, size / near_size)

        reason = None
        if source == "general" and detection.confidence < gate["object_min_confidence"]:
            reason = "low_confidence"
        elif size < min_size:
            reason = "far"
        elif position != "center" and (label == "Object" or size < min_size * gate["side_multiplier"]):
            reason = "side"
        band = "far" if reason == "far" else _band(size, near_size, depth_proximity)

        weight = OBJECT_WEIGHT if label == "Object" else severity[label]
        priority = weight * (0.5 + 0.5 * proximity)
        hazards.append(Hazard(label, proximity, position, priority, xyxy, size=size, band=band,
                              source=source, confidence=detection.confidence,
                              class_id=detection.class_id, gated=reason is None, reason=reason))
    # Any trained hazard outranks any generic Object, whatever their scores.
    return sorted(hazards, key=lambda h: (h.gated, h.label != "Object", h.priority), reverse=True)


def phrase_for(hazard) -> str:
    position = "ahead" if hazard.position == "center" else f"on your {hazard.position}"
    if hazard.label == 'Object':
        return f"Object detected {position}"
    if hazard.band in BAND_WORD:
        return f"{BAND_WORD[hazard.band]}, {hazard.label} {position}"
    if hazard.proximity is None:
        # Gate disabled and no depth: no distance information at all.
        suffix = "" if hazard.label == 'People' else "; distance unavailable"
        return f"Note, {hazard.label} {position}{suffix}"
    urgency = "Warning" if hazard.proximity > 0.7 else "Caution" if hazard.proximity > 0.4 else "Note"
    return f"{urgency}, {hazard.label} {position}"


class AlertCooldown:
    """Tracks when each (label, position, band) hazard was last announced so the
    speech stage does not repeat itself every frame. A hazard re-alerts early when
    it gets meaningfully closer. Keying by band means a far hazard can never mute a
    nearer one at the same position. Generic detections use a longer cooldown."""

    def __init__(self, cooldown_seconds: float = COOLDOWN_SECONDS,
                 general_cooldown_seconds: float | None = None,
                 proximity_jump: float = PROXIMITY_JUMP_TO_REALERT, clock=time.monotonic):
        self.cooldown_seconds = cooldown_seconds
        self.general_cooldown_seconds = (cooldown_seconds if general_cooldown_seconds is None
                                         else general_cooldown_seconds)
        self.proximity_jump = proximity_jump
        self.clock = clock
        self._last: dict[tuple, tuple[float, float | None]] = {}

    @classmethod
    def from_gate(cls, gate=None, clock=time.monotonic):
        gate = ALERT_GATE if gate is None else gate
        return cls(gate["cooldown_seconds"]["trained"], gate["cooldown_seconds"]["Object"],
                   gate["proximity_jump_to_realert"], clock)

    def is_ready(self, hazard: Hazard, now: float | None = None) -> bool:
        """True if announcing now is allowed. Does not record anything."""
        now = self.clock() if now is None else now
        last = self._last.get(hazard.key)
        if last is None:
            return True
        last_time, last_proximity = last
        cooldown = self.general_cooldown_seconds if hazard.source == "general" else self.cooldown_seconds
        got_closer = (hazard.proximity is not None and last_proximity is not None
                      and hazard.proximity - last_proximity > self.proximity_jump)
        return now - last_time > cooldown or got_closer

    def record(self, hazard: Hazard, now: float | None = None):
        self._last[hazard.key] = (self.clock() if now is None else now, hazard.proximity)

    def should_announce(self, hazard: Hazard, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        if self.is_ready(hazard, now):
            self.record(hazard, now)
            return True
        return False
