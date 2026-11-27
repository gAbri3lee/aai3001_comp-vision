"""
Safety decision and response engine. It is rule based by design, not a trained
model. It turns detections into hazards, applies the alert gate (size, path and
confidence rules), ranks what is left and applies a cooldown so the same hazard
is not announced every frame.

The same functions process local and cloud detections, so both backends follow
identical alert rules. Detections are boxes in original-frame pixels.
"""
import time
from collections import deque
from dataclasses import dataclass

from config import load_config
from golden import iou

# Application heuristics; these are not learned model scores.
_CONFIG = load_config()
SEVERITY = _CONFIG["severity"]
ALERT_GATE = _CONFIG["alert_gate"]

COOLDOWN_SECONDS = 4.0
CONFIRM_IOU = 0.2   # A box in an earlier frame counts as the same hazard if it overlaps this much (same label)
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
    source: str = "trained"          # "trained" (active trained model) or "general" (COCO model)
    confidence: float | None = None
    class_id: int | None = None
    gated: bool = True               # True: passed the alert gate and may be announced
    reason: str | None = None        # Why it was ignored: far / side / low_confidence
    index: int | None = None         # Position in the detector's output list

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
    for index, detection in enumerate(detections):
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
                              class_id=detection.class_id, index=index))
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
    for index, detection in enumerate(detections):
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
                              class_id=detection.class_id, gated=reason is None, reason=reason,
                              index=index))
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


@dataclass
class FrameAlerts:
    hazards: list            # Every detection as a Hazard, gated ones first
    alerts: list             # Phrases of the gated hazards, in priority order
    banner: str              # Top gated phrase, or "" when nothing passed the gate
    selected: Hazard | None  # The hazard being spoken (this frame, or still in view)
    speak: str | None        # Phrase to send to the speaker this frame, if any


@dataclass
class PersonEncounter:
    box: tuple
    seen: float
    announced_size: float | None = None
    outside_path: bool = False


def match_people(tracks, hazards, now):
    """Associate each visible person once, including people outside the alert gate."""
    tracks[:] = [track for track in tracks if now - track.seen <= 1.5]
    people = [h for h in hazards if h.label == 'People']
    matches = {}
    used = set()
    candidates = sorted(((iou(track.box, h.xyxy), i, j)
                         for i, track in enumerate(tracks) for j, h in enumerate(people)), reverse=True)
    for overlap, i, j in candidates:
        if overlap < .12:
            break
        if i in used or j in matches:
            continue
        matches[j] = tracks[i]
        used.add(i)
    for j, h in enumerate(people):
        if j not in matches:
            matches[j] = PersonEncounter(h.xyxy, now)
            tracks.append(matches[j])
        track = matches[j]
        track.box, track.seen = h.xyxy, now
    return {id(h): matches[j] for j, h in enumerate(people)}


class AlertPolicy:
    """One alert decision per frame (decision D6), shared by the web app and replay.

    The banner is the top gated hazard. The speaker gets the highest-priority gated
    hazard whose cooldown has expired. Both come from the same frame, so a spoken
    phrase always belongs to a hazard that is currently visible and gated. The
    selected hazard stays marked while the same label and position stay in view.
    """

    def __init__(self, gate=None, severity=None, clock=time.monotonic):
        self.gate = ALERT_GATE if gate is None else gate
        self.severity = SEVERITY if severity is None else severity
        self.clock = clock
        self.cooldown = AlertCooldown.from_gate(self.gate, clock)
        self.people = []
        self._selected = None
        confirm = self.gate.get("confirm_frames") or {"required": 1, "window": 1}
        self.confirm_required = confirm["required"]
        self._history = deque(maxlen=max(confirm["window"] - 1, 0))   # earlier detector frames: [(label, box)]

    def confirmed(self, hazard) -> bool:
        """Frame confirmation: speak a hazard only once the same label has been detected at
        about the same place (box overlap >= CONFIRM_IOU) in at least `required` of the last
        `window` detector frames, this frame included. 1-of-1 (the default) always confirms."""
        if self.confirm_required <= 1:
            return True
        seen = 1 + sum(any(label == hazard.label and iou(box, hazard.xyxy) >= CONFIRM_IOU for label, box in frame)
                       for frame in self._history)
        return seen >= self.confirm_required

    def reset(self):
        """Forget every cooldown and the marked hazard, so the next gated hazard is
        spoken. Used when the detector changes (decision U3); the rules are unchanged."""
        self.cooldown = AlertCooldown.from_gate(self.gate, self.clock)
        self.people = []
        self._selected = None
        self._history.clear()

    def process(self, detections, frame_size, depth_map=None, now=None) -> FrameAlerts:
        now = self.clock() if now is None else now
        hazards = assess(detections, frame_size, self.gate, self.severity, depth_map)
        announceable = [hazard for hazard in hazards if hazard.gated]
        alerts = list(dict.fromkeys(phrase_for(hazard) for hazard in announceable))
        tracks = match_people(self.people, hazards, now)
        left, right = self.gate["centre_third"]
        for hazard in hazards:
            if hazard.label == "People":
                midpoint = (hazard.xyxy[0] + hazard.xyxy[2]) / (2 * frame_size[0])
                if midpoint < left - .04 or midpoint > right + .04:
                    tracks[id(hazard)].outside_path = True
        selected = None
        for hazard in announceable:
            if not self.confirmed(hazard):
                continue
            if hazard.label == 'People':
                track = tracks[id(hazard)]
                size = (hazard.xyxy[3] - hazard.xyxy[1]) / frame_size[1]
                closer = track.announced_size is not None and size >= track.announced_size + max(.08, .25 * track.announced_size)
                entered = hazard.position == 'center' and track.outside_path
                if track.announced_size is not None and not (closer or entered):
                    continue
                track.announced_size = size
                track.outside_path = hazard.position != "center"
            elif not self.cooldown.should_announce(hazard, now):
                continue
            selected = hazard
            break
        speak = phrase_for(selected) if selected is not None else None
        if selected is None and self._selected is not None:
            selected = next((hazard for hazard in announceable
                             if (hazard.label, hazard.position) == self._selected), None)
        self._selected = (selected.label, selected.position) if selected is not None else None
        if self._history.maxlen:
            self._history.append([(hazard.label, tuple(hazard.xyxy)) for hazard in hazards])
        return FrameAlerts(hazards, alerts, alerts[0] if alerts else "", selected, speak)
