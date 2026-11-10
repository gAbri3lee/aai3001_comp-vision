"""VisionAid web UI: live capture, detection, alert gate, speech and page controls.

Local mode never sends a frame anywhere. When the app starts with --endpoint, a
cloud detector is also built and the page can switch to it (POST /mode). A cloud
timeout or error falls back to local, keeps the last good result for at most one
second, and stays on local until a successful "Retry cloud" (decision U2).
"""
import argparse
import base64
from collections import deque
import copy
import io
import math
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from config import add_detector_arguments, detector_settings, resolve_path
from detectors import DetectorError
from scene_detector import SceneDetector as create_detector
from detection import annotate_hazards
from capture import LatestFrameCapture

from decision_engine import AlertPolicy, phrase_for
import distance

# Status fields cleared whenever results expire, fail or the source ends.
EMPTY_RESULT = dict(alert="", alerts=[], detections=[], selected=None, fps=0.,
                    latency_ms=None, round_trip_ms=None)
BACKENDS = ("local", "cloud")
MODEL_NAMES = {"local": "YOLOv8n", "cloud": "RF-DETR Nano"}
# After a cloud failure the last good result stays on screen at most this long.
FALLBACK_HOLD_SECONDS = 1.0

# Plain messages for problems while running (Phase 2 of the second hand-over).
HEALTH_MESSAGES = {
    "camera_missing": ("error", "Camera not found. Check the connection."),
    "cloud_unreachable": ("warning", "Cloud unreachable, using local."),
    "general_unavailable": ("warning", "Extra object detection is off."),
    "speech_unavailable": ("warning", "Speech is unavailable. Alerts show on screen only."),
    "no_video": ("error", "No video. Waiting for camera."),
    "source_ended": ("info", "The video file has ended. Restart the app to play it again."),
    "pipeline_stopped": ("error", "Processing stopped. Restart the app."),
}


# Speech controls (decision U5): words per minute for macOS `say -r`.
RATE_MIN, RATE_MAX, RATE_DEFAULT = 120, 260, 180
TEST_PHRASE = "VisionAid voice test"
CONTROL_KEYS = ("paused", "muted", "rate", "action")
ACTIONS = ("test_voice", "replay_last", "clear_log")
SPOKEN_LOG_SIZE = 10
BAND_LABELS = {"near": "Warning", "mid-close": "Caution", "mid": "Note"}


def priority_order(severity):
    """Classes from the most to the least severe, then generic Objects, which any
    trained hazard outranks. Built from the config so the page never hard-codes it."""
    return sorted(severity, key=lambda label: -severity[label]) + ["Object"]


# Image analysis (decision U4): JPEG or PNG, at most 10 MB, never written to disk.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_PIXELS = 40_000_000
SAMPLE_DIRS = ("demo", "golden/images")
SAMPLE_SUFFIXES = (".jpg", ".jpeg", ".png")


class UploadError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def decode_upload(data: bytes):
    """Validate type, size and dimensions, then decode to a BGR frame."""
    from PIL import Image, UnidentifiedImageError
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("The image is larger than 10 MB.", 413)
    try:
        image = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise UploadError("Upload a JPEG or PNG image.", 415)
    with image:
        if image.format not in ("JPEG", "PNG"):
            raise UploadError("Upload a JPEG or PNG image.", 415)
        width, height = image.size
        if width * height > MAX_UPLOAD_PIXELS:
            raise UploadError("The image has too many pixels.")
        try:
            # A full decode catches truncated or damaged files that OpenCV would
            # silently return as a partial picture.
            image.load()
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
            raise UploadError("Could not read this image. The file may be damaged.")
    # OpenCV decodes the pixels, exactly as the live camera path and golden checks do.
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise UploadError("Could not read this image. The file may be damaged.")
    return frame


def find_samples():
    """Sample images offered on the page: demo/ if it has images, else golden/images.
    Clients choose by id only, never by path."""
    for folder in SAMPLE_DIRS:
        path = resolve_path(folder)
        if path.is_dir():
            images = sorted(p for p in path.iterdir() if p.suffix.lower() in SAMPLE_SUFFIXES)
            if images:
                return {f"sample-{i}": image for i, image in enumerate(images, 1)}
    return {}


class ControlConflict(RuntimeError):
    """A valid request that cannot be done now (for example replay while muted)."""


def health_entry(code):
    severity, message = HEALTH_MESSAGES[code]
    return {"code": code, "severity": severity, "message": message}


class ModeUnavailable(RuntimeError):
    """The requested mode cannot be used right now (no endpoint, or health failed)."""


class StaleFrame(DetectorError):
    """A frame or result was too old to use. Never a reason to leave cloud mode."""


def _selected_dict(hazard):
    if hazard is None:
        return None
    return {"class": hazard.label, "position": hazard.position, "band": hazard.band,
            "source": hazard.source, "phrase": phrase_for(hazard), "box_xyxy": list(hazard.xyxy)}


def backend_settings(args, backend):
    """Settings for one backend. Local-only overrides do not apply to the cloud."""
    args = copy.copy(args)
    args.backend = backend
    if backend == "cloud":
        args.model = args.imgsz = None
    return detector_settings(args)


def decide(frame, batch, policy, depth_estimator=None, check=None, now=None):
    """The one decision path shared by the live loop and image analysis: optional
    depth, then the alert gate, ranking, cooldown and spoken-hazard choice."""
    depth_map = (depth_estimator.predict(frame)
                 if depth_estimator is not None and batch.detections else None)
    if check is not None:
        check()
    return policy.process(batch.detections, (frame.shape[1], frame.shape[0]), depth_map, now=now)


def annotate(frame, result):
    """Normal and debug renderings of one decision."""
    return (annotate_hazards(frame, result.hazards, result.selected),
            annotate_hazards(frame, result.hazards, result.selected, debug=True))


class PipelineState:
    """Owns the background capture/inference loop and exposes thread-safe
    reads for Flask's request handlers (which run on separate threads)."""

    CAMERA_RETRY_SECONDS = 2.0  # A missing or lost camera is reopened this often.
    NO_VIDEO_SECONDS = 3.0      # No frame for this long: clear results, show a message.

    def __init__(self, args):
        self.args = args
        # Validates the command line exactly as before (for example --model with cloud).
        self.settings = detector_settings(args)
        self.backend_settings = {"local": backend_settings(args, "local")}
        if args.endpoint:
            self.backend_settings["cloud"] = backend_settings(args, "cloud")
        elif args.backend == "cloud":
            raise ValueError("Cloud mode requires --endpoint http(s)://host:port")
        local = create_detector(self.backend_settings["local"])
        self.detectors = {"local": local}
        if "cloud" in self.backend_settings:
            # One general model serves both backends.
            self.detectors["cloud"] = create_detector(self.backend_settings["cloud"],
                                                      general=getattr(local, "general", None))
        if not args.no_depth:
            from depth import DepthEstimator
        if not args.no_tts:
            from tts import Speaker
        self.depth_estimator = None if args.no_depth else DepthEstimator()
        self.speaker = None
        self.speech_error = None
        if not args.no_tts:
            try:
                self.speaker = Speaker()
            except Exception as exc:
                # Alerts still show on screen; the page says speech is unavailable.
                self.speech_error = f"Speech unavailable: {exc}"
        self.paused = False
        self.muted = False
        self.speech_rate = RATE_DEFAULT
        self.last_spoken = None
        self.spoken_log = deque(maxlen=SPOKEN_LOG_SIZE)  # memory only, newest first
        self._paused_shown = False
        if self.speaker is not None:
            self.speaker.rate = RATE_DEFAULT
        self.policy = AlertPolicy(self.settings['alert_gate'], self.settings['severity'])
        # Optional distance estimate for the debug panel only (never spoken).
        self.calibration = distance.load_calibration()
        self._lock = threading.Lock()
        # One inference at a time across the live loop and image analysis.
        self._detect_lock = threading.Lock()
        self.samples = find_samples()
        self._latest_jpeg = None
        self._latest_debug_jpeg = None
        self._frame_captured_at = None
        self.mode = args.backend
        self.fallback_active = False
        self.last_error = ""
        self._reset_policy = False
        self._hold_until = None
        gate = self.settings['alert_gate']
        self.status = {"fps": 0.0, "alert": "", "alerts": [], "selected": None, "spoken_at": None,
                       "error": "", "latency_ms": None, "round_trip_ms": None, "detections": [],
                       "gate_enabled": gate['enabled'], "gate_calibrated": gate.get('calibrated', False),
                       "depth_enabled": not args.no_depth, "tts_enabled": not args.no_tts,
                       "distance_calibrated": self.calibration is not None}

        self._stop = threading.Event()

        self._source = int(args.source) if str(args.source).isdigit() else args.source
        self._is_camera = isinstance(self._source, int)
        self.capture = None
        self.camera_missing = False
        self.source_ended = False
        self.pipeline_stopped = False
        self._next_camera_try = 0.
        self._last_frame_at = time.monotonic()
        try:
            self.capture = LatestFrameCapture(self._source)
        except Exception:
            if not self._is_camera:
                raise  # A missing video file is a startup configuration error.
            self._camera_lost()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # Mode (Local/Cloud) ---------------------------------------------------

    def _mode_fields(self):
        """Caller holds the lock."""
        general = getattr(self.detectors[self.mode], "general", True) is not None
        return {"mode": self.mode, "backend": self.mode,
                "mode_label": f"{self.mode.upper()}, {MODEL_NAMES[self.mode]}",
                "model": self.backend_settings[self.mode]["model"] + (" + general-yolov8n" if general else ""),
                "cloud_available": "cloud" in self.detectors,
                "cloud_unavailable_reason": None if "cloud" in self.detectors else
                    "Cloud unavailable: start the app with --endpoint",
                "fallback_active": self.fallback_active, "last_error": self.last_error}

    def mode_state(self) -> dict:
        with self._lock:
            return self._mode_fields()

    def set_mode(self, backend) -> dict:
        """Switch detector. Cloud is checked first through its health endpoint, which
        sends no frame; a failed check leaves the mode unchanged."""
        if backend not in BACKENDS:
            raise ValueError('backend must be "local" or "cloud"')
        if backend == "cloud":
            if "cloud" not in self.detectors:
                raise ModeUnavailable("Cloud unavailable: start the app with --endpoint")
            try:
                self.detectors["cloud"].health()
            except Exception as exc:
                with self._lock:
                    self.last_error = f"Cloud unreachable: {exc}"
                raise ModeUnavailable(f"Cloud unreachable: {exc}") from exc
        with self._lock:
            if backend != self.mode:
                # Decision U3: the first alert after a switch is spoken.
                self._reset_policy = True
            self.mode = backend
            self.fallback_active = False
            if backend == "cloud":
                self.last_error = ""
            return self._mode_fields()

    def _fall_back(self, exc):
        """Cloud failed: switch to local and hold the last good result briefly."""
        with self._lock:
            self.mode = "local"
            self.fallback_active = True
            self.last_error = f"Cloud unreachable: {exc}"
            self._reset_policy = True
            if self._latest_jpeg is not None and self._hold_until is None:
                self._hold_until = time.monotonic() + FALLBACK_HOLD_SECONDS

    # Reads for the request handlers ---------------------------------------

    def _frame_is_stale(self):
        return (self._frame_captured_at is not None and
                time.monotonic() - self._frame_captured_at > self.args.max_frame_age)

    def _hold_expired(self):
        return self._hold_until is not None and time.monotonic() > self._hold_until

    def get_jpeg(self, debug=False):
        with self._lock:
            if self._frame_is_stale() or self._hold_expired() or self.camera_missing or self._no_video():
                return None
            return self._latest_debug_jpeg if debug else self._latest_jpeg

    def _no_video(self):
        """Caller holds the lock. The camera is open but has sent nothing for a while."""
        return (self.capture is not None and not self.source_ended and not self.pipeline_stopped
                and time.monotonic() - self._last_frame_at > self.NO_VIDEO_SECONDS)

    def _health(self):
        """Caller holds the lock. Plain messages for problems while running."""
        codes = []
        if self.pipeline_stopped:
            codes.append("pipeline_stopped")
        if self.camera_missing:
            codes.append("camera_missing")
        elif self._no_video():
            codes.append("no_video")
        if self.source_ended:
            codes.append("source_ended")
        if self.fallback_active:
            codes.append("cloud_unreachable")
        if getattr(self.detectors[self.mode], "general_error", None):
            codes.append("general_unavailable")
        if not self.args.no_tts and (self.speech_error or getattr(self.speaker, "error", None)):
            codes.append("speech_unavailable")
        return [health_entry(code) for code in codes]

    def get_status(self) -> dict:
        with self._lock:
            status = dict(self.status)
            status.update(self._mode_fields())
            status.update(self._control_fields())
            if self.paused or self._hold_expired() or self.camera_missing or self._no_video():
                status.update(EMPTY_RESULT)
            elif self._frame_is_stale():
                status.update(EMPTY_RESULT, error="Frame is stale; waiting for fresh inference")
            status["health"] = self._health()
            return status

    # Pause, mute and speech ------------------------------------------------

    def _control_fields(self):
        """Caller holds the lock."""
        return {"paused": self.paused, "muted": self.muted, "speech_rate": self.speech_rate,
                "speech_rate_min": RATE_MIN, "speech_rate_max": RATE_MAX,
                "speech_enabled": self.speaker is not None, "last_spoken": self.last_spoken,
                "spoken_log": list(self.spoken_log)}

    def control_state(self) -> dict:
        with self._lock:
            return self._control_fields()

    def control(self, body) -> dict:
        """Apply pause, mute, rate and one-off speech actions. Everything is
        validated before anything changes."""
        if not isinstance(body, dict) or not body:
            raise ValueError("Send a JSON object with paused, muted, rate or action")
        unknown = sorted(set(body) - set(CONTROL_KEYS))
        if unknown:
            raise ValueError(f"Unknown control: {', '.join(unknown)}")
        for key in ("paused", "muted"):
            if key in body and not isinstance(body[key], bool):
                raise ValueError(f"{key} must be true or false")
        rate = body.get("rate")
        if "rate" in body:
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate):
                raise ValueError("rate must be a number of words per minute")
            rate = int(round(min(RATE_MAX, max(RATE_MIN, rate))))
        action = body.get("action")
        if action is not None and action not in ACTIONS:
            raise ValueError(f"action must be one of: {', '.join(ACTIONS)}")

        stop_speech = False
        with self._lock:
            if "paused" in body:
                if body["paused"] != self.paused:
                    self._paused_shown = False
                self.paused = body["paused"]
                stop_speech |= self.paused
                if self.paused:
                    self.status.update(EMPTY_RESULT)
            if "muted" in body:
                self.muted = body["muted"]
                stop_speech |= self.muted
            if rate is not None:
                self.speech_rate = rate
            speaker = self.speaker
        if speaker is not None:
            speaker.rate = self.speech_rate
            if stop_speech:
                speaker.stop()  # mute and pause silence the current phrase at once
        if action == "clear_log":
            with self._lock:
                self.spoken_log.clear()
        elif action is not None:
            self._speech_action(action)
        return self.control_state()

    def _speech_action(self, action):
        with self._lock:
            if self.speaker is None:
                raise ControlConflict("Speech is off. Start the app without --no-tts.")
            if self.muted:
                raise ControlConflict("Speech is muted. Unmute first.")
            text = TEST_PHRASE if action == "test_voice" else self.last_spoken
            if text is None:
                raise ControlConflict("No alert has been spoken yet.")
        self._speak(text, kind="test" if action == "test_voice" else "replay")

    def _speak(self, text, kind="alert"):
        """The only way speech is sent. Returns True if the phrase went to the speaker."""
        with self._lock:
            if self.speaker is None or self.muted or (kind == "alert" and self.paused):
                return False
            if kind == "alert":
                self.last_spoken = text
            self.spoken_log.appendleft({"time": time.time(), "phrase": text, "backend": self.mode, "kind": kind})
            speaker = self.speaker
        speaker.say(text)
        return True

    def _show_paused(self, frame):
        """Keep the last frame on screen with a Paused label; no inference."""
        with self._lock:
            if self._paused_shown:
                return
            self._paused_shown = True
        shown = frame.copy()
        height, width = shown.shape[:2]
        cv2.rectangle(shown, (0, 0), (width, 56), (20, 20, 20), -1)
        cv2.putText(shown, "Paused", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        ok, buffer = cv2.imencode(".jpg", shown)
        with self._lock:
            self.status.update(EMPTY_RESULT)
            if ok:
                self._latest_jpeg = self._latest_debug_jpeg = buffer.tobytes()
                self._frame_captured_at = None  # a paused frame is never "stale"

    # Image analysis -----------------------------------------------------

    def analyze(self, frame, source, speak=False, debug=False) -> dict:
        """Run one image through the live path: the active detector, decide() and
        annotate(). A fresh AlertPolicy gives analysis its own cooldown state, so it
        never disturbs the live loop and always selects the top gated hazard."""
        with self._lock:
            mode = self.mode
            model = self._mode_fields()["model"]
        policy = AlertPolicy(self.settings['alert_gate'], self.settings['severity'])
        with self._detect_lock:
            batch = self.detectors[mode].predict(frame)
            result = decide(frame, batch, policy, self.depth_estimator)
        normal, debug_image = annotate(frame, result)
        ok, buffer = cv2.imencode(".jpg", debug_image if debug else normal)
        if not ok:
            raise RuntimeError("Could not encode the annotated image")
        hazards = sorted(result.hazards, key=lambda h: h.index)
        spoken = bool(speak and result.speak and self._speak(result.speak, kind="upload"))
        return {"source": source, "mode": mode, "model": model,
                "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                "alert": result.banner, "alerts": result.alerts,
                "selected": _selected_dict(result.selected),
                "detections": [self._detection_dict(h, frame.shape[0]) for h in hazards],
                "latency_ms": batch.latency_ms, "round_trip_ms": batch.round_trip_ms,
                "spoken": spoken,
                "image": "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")}

    def _detection_dict(self, hazard, frame_height):
        item = hazard.to_dict()
        if self.calibration is not None and hazard.label in self.settings['alert_gate']['known_heights_m']:
            value, note = distance.estimate(hazard.label, hazard.xyxy, frame_height, self.calibration,
                                            self.settings['alert_gate']['known_heights_m'])
            item['distance_est_m'] = None if value is None else round(value, 1)
            item['distance_note'] = note
        return item

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.args.timeout + 3)
        if self.capture is not None:
            self.capture.release()

    # Camera ---------------------------------------------------------------

    def _camera_lost(self):
        """Forget the camera and its results; the loop reopens it later."""
        if self.capture is not None:
            self.capture.release()
        with self._lock:
            self.capture = None
            self.camera_missing = True
            self.status.update(EMPTY_RESULT)
            self._latest_jpeg = self._latest_debug_jpeg = None
            self._frame_captured_at = None
        self._next_camera_try = time.monotonic() + self.CAMERA_RETRY_SECONDS

    def _try_open_camera(self):
        if time.monotonic() < self._next_camera_try:
            return False
        try:
            capture = LatestFrameCapture(self._source)
        except Exception:
            self._next_camera_try = time.monotonic() + self.CAMERA_RETRY_SECONDS
            return False
        with self._lock:
            self.capture = capture
            self.camera_missing = False
            self._last_frame_at = time.monotonic()
        return True

    # Background loop ------------------------------------------------------

    def _loop(self):
        try:
            prev_time = time.monotonic()
            while not self._stop.is_set():
                if self.capture is None:
                    if not self._try_open_camera():
                        self._stop.wait(0.05)
                    continue
                ok, frame, captured_at = self.capture.read()
                if not ok:
                    if self.capture.finished:
                        if self._is_camera:
                            self._camera_lost()  # Unplugged: keep the page up and retry.
                            continue
                        with self._lock:
                            self.status.update(EMPTY_RESULT, error="Camera/source ended")
                            self.source_ended = True
                            self._latest_jpeg = self._latest_debug_jpeg = None
                            self._frame_captured_at = None
                        break
                    self._stop.wait(0.01)
                    continue
                with self._lock:
                    self._last_frame_at = time.monotonic()
                    paused = self.paused
                    mode = self.mode
                    if self._reset_policy:
                        self.policy.reset()
                        self._reset_policy = False
                if paused:
                    self._show_paused(frame)  # the camera stays open; nothing is inferred
                    continue
                error = ""
                batch = None
                result = None
                spoken_at = None
                annotated = debug_annotated = frame

                def check_fresh(what):
                    if time.monotonic() - captured_at > self.args.max_frame_age:
                        raise StaleFrame(what)
                try:
                    check_fresh('Captured frame is stale; discarded')
                    with self._detect_lock:
                        batch = self.detectors[mode].predict(frame)
                except StaleFrame as exc:
                    error = str(exc)
                except Exception as exc:
                    if mode == "cloud":
                        self._fall_back(exc)
                        continue
                    error = str(exc)
                if batch is not None:
                    try:
                        # One decision per frame: banner, list, highlight and speech all
                        # come from these gated hazards (decision D6).
                        with self._detect_lock:
                            result = decide(frame, batch, self.policy, self.depth_estimator,
                                            check=lambda: check_fresh('Inference result is stale; discarded'))
                        annotated, debug_annotated = annotate(frame, result)
                        if result.speak is not None and self._speak(result.speak):
                            spoken_at = time.time()
                    except Exception as exc:
                        # Never retain prior boxes/alerts after an inference failure.
                        error = str(exc)
                if error:
                    batch = result = None
                    annotated = debug_annotated = frame

                now = time.monotonic()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                ok_jpeg, buffer = cv2.imencode(".jpg", annotated)
                ok_debug, debug_buffer = cv2.imencode(".jpg", debug_annotated)
                hazards = sorted(result.hazards, key=lambda h: h.index) if result else []
                with self._lock:
                    self._hold_until = None
                    if ok_jpeg and ok_debug:
                        self._latest_jpeg = buffer.tobytes()
                        self._latest_debug_jpeg = debug_buffer.tobytes()
                        self._frame_captured_at = captured_at
                    self.status["fps"] = round(fps, 1)
                    self.status["alert"] = result.banner if result else ""
                    self.status["alerts"] = result.alerts if result else []
                    self.status["selected"] = _selected_dict(result.selected) if result else None
                    if spoken_at is not None:
                        self.status["spoken_at"] = spoken_at
                    self.status["error"] = error
                    self.status["latency_ms"] = batch.latency_ms if batch else None
                    self.status["round_trip_ms"] = batch.round_trip_ms if batch else None
                    self.status["detections"] = [self._detection_dict(hazard, frame.shape[0]) for hazard in hazards]
        except Exception as exc:
            with self._lock:
                self.status['error'] = f'Pipeline stopped: {exc}'
                self.pipeline_stopped = True
        finally:
            if self.capture is not None:
                self.capture.release()
            with self._lock:
                self.status.update(EMPTY_RESULT)
                if not self.status['error']:
                    self.status['error'] = 'Pipeline stopped'
                self._latest_jpeg = self._latest_debug_jpeg = None
                self._frame_captured_at = None


def create_app(state: PipelineState) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        # ?debug=1 also draws ignored (far, side, low-confidence) boxes and sizes.
        debug = request.args.get("debug") == "1"

        def generate():
            while True:
                jpeg = state.get_jpeg(debug)
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.03)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        return jsonify(state.get_status())

    @app.post("/control")
    def control():
        try:
            return jsonify(state.control(request.get_json(silent=True)))
        except ControlConflict as exc:
            return jsonify(error=str(exc), **state.control_state()), 409
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    # Multipart overhead on top of the 10 MB image limit; the exact limit is checked later.
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 256 * 1024

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="The image is larger than 10 MB."), 413

    @app.get("/config")
    def config():
        """What the page needs to draw the legend and controls, from the config."""
        return jsonify(priority=priority_order(state.settings["severity"]), band_labels=BAND_LABELS,
                       speech_rate_min=RATE_MIN, speech_rate_max=RATE_MAX, speech_rate_default=RATE_DEFAULT,
                       shortcuts={"Space": "pause", "M": "mute", "C": "mode"},
                       max_upload_bytes=MAX_UPLOAD_BYTES)

    @app.get("/samples")
    def samples():
        return jsonify([{"id": key, "name": f"Sample {i}"}
                        for i, key in enumerate(state.samples, 1)])

    @app.post("/analyze")
    def analyze():
        """Multipart field `image` (JPEG or PNG) or form field `sample` (an id from
        /samples). Optional `speak=1` and `debug=1`. Nothing is stored."""
        form = request.form
        speak = form.get("speak") in ("1", "true", "on")
        debug = form.get("debug") in ("1", "true", "on")
        try:
            if "image" in request.files:
                frame, source = decode_upload(request.files["image"].read()), "upload"
            elif form.get("sample"):
                path = state.samples.get(form["sample"])
                if path is None:
                    raise UploadError("Unknown sample.", 404)
                frame, source = decode_upload(path.read_bytes()), f"Sample {list(state.samples).index(form['sample']) + 1}"
            else:
                raise UploadError("Choose an image file or a sample.")
        except UploadError as exc:
            return jsonify(error=str(exc)), exc.status
        try:
            return jsonify(state.analyze(frame, source, speak=speak, debug=debug))
        except Exception as exc:
            return jsonify(error=f"Analysis failed: {exc}"), 503

    @app.post("/mode")
    def mode():
        body = request.get_json(silent=True)
        backend = body.get("backend") if isinstance(body, dict) else None
        try:
            return jsonify(state.set_mode(backend))
        except ModeUnavailable as exc:
            return jsonify(error=str(exc), **state.mode_state()), 503
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="VisionAid web UI")
    parser.add_argument("--source", default=0, help="Camera index or video file path")
    add_detector_arguments(parser)
    parser.add_argument("--no-depth", action="store_true", help="Disable depth estimation stage")
    parser.add_argument("--no-tts", action="store_true", help="Disable spoken alerts")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    state = PipelineState(args)
    app = create_app(state)
    print(f"VisionAid web UI running at http://{args.host}:{args.port}")
    if args.endpoint:
        print(f"Cloud mode available ({args.endpoint}); starting in {args.backend} mode")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        state.stop()


if __name__ == "__main__":
    main()
