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
from config import CLASS_NAMES, add_detector_arguments, detector_settings, load_config, resolve_path
from detectors import DetectorError
from scene_detector import SceneDetector as create_detector
from detection import annotate_hazards, annotate_recognition
from capture import LatestFrameCapture, limit_size
import media

from decision_engine import AlertPolicy, assess, phrase_for
import distance

# Status fields cleared whenever results expire, fail or the source ends.
EMPTY_RESULT = dict(alert="", alerts=[], detections=[], recognition=[], selected=None, fps=0.,
                    latency_ms=None, round_trip_ms=None, pipeline_latency_ms=None,
                    processing_ms=None, preview_encode_ms=None)
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
    "video_ended": ("info", "The video has ended. Replay it or go back to the camera."),
    "video_deleted": ("info", "The video has ended and its temporary copy was deleted. Upload it again to replay it."),
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
# Up to 20 images per request (decision V4).
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_BYTES = media.MAX_BATCH * MAX_UPLOAD_BYTES + 1024 * 1024
# Recognition view: every trained detection, no gate, no speech. Alert view: the live rules.
VIEWS = ("recognition", "alert")
MAX_UPLOAD_PIXELS = 40_000_000
SAMPLE_DIRS = ("demo", "golden/images")
SAMPLE_SUFFIXES = (".jpg", ".jpeg", ".png")


UploadError = media.MediaError  # a rejected file: message for the page plus an HTTP status


def decode_upload(data: bytes):
    """Validate type, size and dimensions, then decode to a BGR frame."""
    from PIL import Image, UnidentifiedImageError
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("The image is larger than 10 MB.", 413)
    if not (data.startswith(b'\xff\xd8\xff') or data.startswith(b'\x89PNG\r\n\x1a\n')):
        raise UploadError("Upload a JPEG or PNG image.", 415)
    try:
        image = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ImportError, Image.DecompressionBombError):
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
                samples = {f"sample-{i}": image for i, image in enumerate(images, 1)}
                for obstacle in sorted(resolve_path("demo/obstacles").glob("*.jpg")):
                    samples["object-" + obstacle.stem] = obstacle
                return samples
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


def jpeg_data_url(image):
    """An annotated image for the page, at most MAX_DISPLAY_SIDE on its long side."""
    ok, buffer = cv2.imencode(".jpg", limit_size(image, media.MAX_DISPLAY_SIDE),
                              [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Could not encode the annotated image")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def form_flag(form, name):
    return form.get(name) in ("1", "true", "on")


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
        config = load_config(args.config)
        self.accuracy_note = media.accuracy_note(config)
        self.video_accuracy_note = media.accuracy_note(config, "this video")
        # Optional distance estimate for the debug panel only (never spoken).
        self.calibration = distance.load_calibration()
        self._lock = threading.Lock()
        # One inference at a time across the live loop and image analysis.
        self._detect_lock = threading.Lock()
        self.samples = find_samples()
        self.demo_root = resolve_path(media.DEMO_DIR)
        self._latest_jpeg = None
        self._latest_debug_jpeg = None
        self._frame_captured_at = None
        self.mode = args.backend
        self.fallback_active = False
        self.last_error = ""
        self._reset_policy = False
        self._generation = 0  # Invalidates work started before a control/source change.
        self._hold_until = None
        gate = self.settings['alert_gate']
        self.status = {"fps": 0.0, "alert": "", "alerts": [], "selected": None, "spoken_at": None,
                       "error": "", "latency_ms": None, "round_trip_ms": None, "detections": [],
                       "gate_enabled": gate['enabled'], "gate_calibrated": gate.get('calibrated', False),
                       "depth_enabled": not args.no_depth, "tts_enabled": not args.no_tts,
                       "distance_calibrated": self.calibration is not None}

        self._stop = threading.Event()

        # Uploaded or library video playing instead of the start source (third hand-over).
        media.TempStore.remove_stale()
        self.store = media.TempStore()
        self.video = None
        self.video_ended = False
        self._video_ended_at = None
        self.video_position = 0.0
        self.view = "alert"
        self.summary = media.ClassSummary()
        self._pending = None
        self._last_frame = None
        self._upload_lock = threading.Lock()
        # Decoder opening and source disposal share one transaction. Otherwise a
        # camera switch can delete a file while Replay is still opening it.
        self._source_io_lock = threading.RLock()

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

    def _model_label(self, mode):
        general = getattr(self.detectors[mode], "general", True) is not None
        return self.backend_settings[mode]["model"] + (" + general-yolov8n" if general else "")

    def _mode_fields(self):
        """Caller holds the lock."""
        return {"mode": self.mode, "backend": self.mode,
                "mode_label": f"{self.mode.upper()}, {MODEL_NAMES[self.mode]}",
                "model": self._model_label(self.mode),
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
                self._generation += 1
                self.status.update(EMPTY_RESULT)
                self._latest_jpeg = self._latest_debug_jpeg = None
                self._frame_captured_at = None
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
            self._generation += 1
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
                and not self.paused and time.monotonic() - self._last_frame_at > self.NO_VIDEO_SECONDS)

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
        if self.video is not None and self.video_ended:
            codes.append("video_ended" if self.video["path"].exists() else "video_deleted")
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
            status["source"] = self._source_fields()
            status["view"] = self.view
            status["video_summary"] = self.summary.to_list() if self.video is not None else None
            status["frames_processed"] = self.summary.frames if self.video is not None else None
            status["accuracy_note"] = self.video_accuracy_note
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
            capture = self.capture
            file_source = self.video is not None or not self._is_camera
            if "paused" in body:
                if body["paused"] != self.paused:
                    self._paused_shown = False
                    self._generation += 1
                    self._reset_policy = True
                self.paused = body["paused"]
                stop_speech |= self.paused
                if self.paused:
                    self.status.update(EMPTY_RESULT)
            if "muted" in body:
                if self.muted and not body["muted"]:
                    self._reset_policy = True
                self.muted = body["muted"]
                stop_speech |= self.muted
            if rate is not None:
                self.speech_rate = rate
            speaker = self.speaker
        if "paused" in body and file_source and capture is not None:
            # A video stops where it is; a camera keeps streaming while paused.
            toggle = getattr(capture, "pause" if body["paused"] else "resume", None)
            if toggle is not None:
                toggle()
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

    def _speak(self, text, kind="alert", generation=None):
        """The only way speech is sent. Returns True if the phrase went to the speaker."""
        with self._lock:
            if (self.speaker is None or self.muted or self._stop.is_set()
                    or (kind == "alert" and self.paused)
                    or (generation is not None and generation != self._generation)):
                return False
            # Queueing is nonblocking; keep it ordered with mute/pause changes.
            if not self.speaker.say(text, requested=kind != "alert"):
                return False
            if kind in ("alert", "upload"):
                self.last_spoken = text
            self.spoken_log.appendleft({"time": time.time(), "phrase": text, "backend": self.mode, "kind": kind})
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

    def analyze(self, frame, source, speak=False, debug=False, view="alert", expected=None,
                mode=None) -> dict:
        """Run one image through the live path: the detector, decide() and annotate().
        A fresh AlertPolicy gives analysis its own cooldown state, so it never
        disturbs the live loop and always selects the top gated hazard.

        Both views come from the same detections. Recognition view lists every
        trained detection (no gate) and never speaks. In Alert view the result is
        read aloud when asked for and not muted, even if no hazard passed the gate."""
        with self._lock:
            mode = mode or self.mode
            model = self._model_label(mode)
        policy = AlertPolicy(self.settings['alert_gate'], self.settings['severity'])
        with self._detect_lock:
            batch = self.detectors[mode].predict(frame)
            result = decide(frame, batch, policy, self.depth_estimator)
        normal, debug_image = annotate(frame, result)
        hazards = sorted(result.hazards, key=lambda h: h.index)
        recognition = media.recognition_dicts(result.hazards)
        images = {"alert": jpeg_data_url(debug_image if debug else normal),
                  "recognition": jpeg_data_url(annotate_recognition(frame, result.hazards))}
        spoken = bool(view == "alert" and speak and
                      self._speak(result.speak or "No hazards present", kind="upload"))
        output = {"source": source, "mode": mode, "model": model, "view": view,
                  "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                  "alert": result.banner, "alerts": result.alerts,
                  "selected": _selected_dict(result.selected),
                  "detections": [self._detection_dict(h, frame.shape[0]) for h in hazards],
                  "recognition": recognition,
                  "latency_ms": batch.latency_ms, "round_trip_ms": batch.round_trip_ms,
                  "spoken": spoken, "images": images, "image": images[view]}
        if expected is not None:
            output["expected"] = media.expected_result(recognition, expected)
        return output

    def analyze_batch(self, items, view="alert", expected=None, compare=False, speak=False,
                      debug=False) -> dict:
        """Analyse up to MAX_BATCH images. `items` holds (name, source, frame, error);
        a file that failed validation keeps its error and never stops the others.

        Images go to the cloud only in cloud mode or when compare is asked for
        (decision V6). Speech is used only for a single image in Alert view."""
        if compare and "cloud" not in self.detectors:
            raise ModeUnavailable("Compare needs the cloud. Start the app with --endpoint.")
        with self._lock:
            backends = list(BACKENDS) if compare else [self.mode]
        speak = speak and len(items) == 1 and not compare
        results = []
        for index, (name, source, frame, error) in enumerate(items):
            entry = {"index": index, "name": name, "source": source}
            if error is not None:
                entry["error"] = error
                results.append(entry)
                continue
            runs = []
            for backend in backends:
                try:
                    run = self.analyze(frame, source, speak=speak, debug=debug, view=view,
                                       expected=expected, mode=backend)
                    del run["image"]  # `images` holds both views; keep the batch small
                    runs.append(run)
                except Exception as exc:
                    runs.append({"mode": backend, "model": self._model_label(backend),
                                 "error": f"Analysis failed: {exc}"})
            entry["runs"] = runs
            results.append(entry)
        summary = [media.batch_summary(results, backend, self._model_label(backend), expected)
                   for backend in backends] if expected else []
        return {"view": view, "expected_class": expected, "compare": compare, "backends": backends,
                "cloud_used": "cloud" in backends, "results": results, "summary": summary,
                "accuracy_note": self.accuracy_note}

    def _detection_dict(self, hazard, frame_height, *, calibration_applicable=False):
        item = hazard.to_dict()
        if self.calibration is not None and hazard.label in self.settings['alert_gate']['known_heights_m']:
            if calibration_applicable:
                value, note = distance.estimate(hazard.label, hazard.xyxy, frame_height, self.calibration,
                                                self.settings['alert_gate']['known_heights_m'])
            else:
                value, note = None, 'live camera calibration does not apply to uploaded/file media'
            item['distance_est_m'] = None if value is None else round(value, 1)
            item['distance_note'] = note
        return item

    # Video source ---------------------------------------------------------

    VIDEO_KEEP_SECONDS = 600  # an ended upload stays available for Replay this long

    def _source_fields(self):
        """Caller holds the lock. The name is for display only, never a path."""
        home = "camera" if self._is_camera else "file"
        # A switch the loop has not applied yet is reported as already started.
        pending = self._pending is not None
        video = self._pending[1] if pending else self.video
        if video is None:
            return {"type": "camera" if self._is_camera else "video", "origin": home, "home": home,
                    "name": "Camera" if self._is_camera else "Video file", "duration": None,
                    "position": None, "fps": None, "ended": False if pending else self.source_ended,
                    "replay_available": False}
        position = 0.0 if pending else min(self.video_position, video["duration"])
        return {"type": "video", "origin": "upload" if video["temporary"] else "library", "home": home,
                "name": video["name"], "duration": video["duration"], "position": round(position, 1),
                "fps": video["fps"], "ended": False if pending else self.video_ended,
                "replay_available": video["path"].exists()}

    def source_state(self) -> dict:
        with self._lock:
            return {"source": self._source_fields(), "view": self.view}

    def upload_video(self, file) -> dict:
        """Save one uploaded video under a random private name, check it and play it
        through the live loop. One upload at a time; any failure deletes the copy."""
        if not self._upload_lock.acquire(blocking=False):
            raise UploadError("Another video is being uploaded. Wait for it to finish.", 409)
        try:
            name = media.display_name(file.filename, "Uploaded video")
            path = self.store.save(file.stream, media.video_suffix(name))
            try:
                self._play(path, name, temporary=True)
            except BaseException:
                self.store.delete(path)
                raise
        finally:
            self._upload_lock.release()
        return self.source_state()

    def play_library_video(self, path, name) -> dict:
        """A prepared video from demo/videos. It is the presenter's file: never deleted."""
        self._play(path, name, temporary=False)
        return self.source_state()

    def _play(self, path, name, temporary):
        with self._source_io_lock:
            return self._play_locked(path, name, temporary)

    def _play_locked(self, path, name, temporary):
        info = media.probe_video(path)
        try:
            capture = LatestFrameCapture(str(path), max_side=media.INFERENCE_MAX_SIDE)
        except RuntimeError as exc:
            raise UploadError("Could not read this video. The file may be damaged.") from exc
        self._switch(capture, {"name": name, "path": path, "temporary": temporary, **info})

    def source_control(self, body) -> dict:
        with self._source_io_lock:
            return self._source_control_locked(body)

    def _source_control_locked(self, body) -> dict:
        """`action`: restart (also Replay) or camera (back to the start source).
        `view`: recognition or alert. Validated before anything changes."""
        if not isinstance(body, dict) or not body:
            raise ValueError("Send a JSON object with action or view")
        unknown = sorted(set(body) - {"action", "view"})
        if unknown:
            raise ValueError(f"Unknown control: {', '.join(unknown)}")
        action, view = body.get("action"), body.get("view")
        if action is not None and action not in ("restart", "camera"):
            raise ValueError('action must be "restart" or "camera"')
        if view is not None and view not in VIEWS:
            raise ValueError('view must be "recognition" or "alert"')
        if view is not None:
            with self._lock:
                if view != self.view:
                    self._reset_policy = True  # back in Alert view, the next hazard is spoken
                    self._generation += 1
                self.view = view
                speaker = self.speaker
            if view == "recognition" and speaker is not None:
                speaker.stop()  # Recognition view never speaks
        if action == "restart":
            with self._lock:
                video = self.video
            if video is None:
                raise ControlConflict("No video is playing.")
            if not video["path"].exists():
                raise ControlConflict("The temporary copy was deleted. Upload the video again.")
            try:
                capture = LatestFrameCapture(str(video["path"]), max_side=media.INFERENCE_MAX_SIDE)
            except RuntimeError as exc:
                raise ControlConflict("Could not reopen the video.") from exc
            self._switch(capture, video)
        elif action == "camera":
            self._switch(None, None)
        return self.source_state()

    def _switch(self, capture, video):
        with self._source_io_lock:
            return self._switch_locked(capture, video)

    def _switch_locked(self, capture, video):
        """Hand a new source to the loop thread, which swaps it in between frames.
        A switch that was still waiting is dropped and its file deleted."""
        with self._lock:
            replaced, self._pending = self._pending, (capture, video)
            active = self.video
            self._generation += 1
            if video is None:
                self.view = "alert"  # the camera always uses the live rules
        if replaced is not None:
            # A cancelled Replay can share the active video's file. Release its
            # new decoder now; delete only after the loop retires the active one.
            keep = active if (active is not None and replaced[1] is not None
                              and active['path'] == replaced[1]['path']) else video
            self._discard(*replaced, keep=keep)

    def _discard(self, capture, video, keep=None):
        if capture is not None:
            capture.release()
        if video is not None and video["temporary"] and (keep is None or keep["path"] != video["path"]):
            self.store.delete(video["path"])

    def _apply_pending(self):
        with self._source_io_lock:
            return self._apply_pending_locked()

    def _apply_pending_locked(self):
        """Loop thread only: start the source asked for by upload, Replay or Back to camera."""
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is None:
            return
        capture, video = pending
        old_capture, old_video = self.capture, self.video
        self._discard(old_capture if old_capture is not capture else None, old_video, keep=video)
        with self._lock:
            self.capture, self.video = capture, video
            self.video_ended, self._video_ended_at, self.video_position = False, None, 0.0
            self.source_ended = self.camera_missing = False
            self.summary = media.ClassSummary()
            self._reset_policy = True  # a new source starts with no cooldowns
            if video is None:
                self.view = "alert"  # the camera always uses the live rules
            self.status.update(EMPTY_RESULT, error="")
            self._latest_jpeg = self._latest_debug_jpeg = None
            self._frame_captured_at = None
            self._last_frame, self._paused_shown = None, False
            self._last_frame_at = time.monotonic()
            paused = self.paused
        if capture is not None and paused:
            capture.pause()
        if capture is None:
            self._next_camera_try = 0.

    def _video_finished(self):
        """Decision V8: the pipeline stays up; the page shows the summary, Replay
        and Back to camera. The last picture stays on screen."""
        self.capture.release()
        with self._lock:
            self.capture = None
            self.video_ended, self._video_ended_at = True, time.monotonic()
            self.status.update(EMPTY_RESULT)
            self._frame_captured_at = None  # the last picture is kept, never "stale"

    def _expire_ended_video(self):
        with self._lock:
            video, ended_at = self.video, self._video_ended_at
        if (video is not None and video["temporary"] and ended_at is not None
                and time.monotonic() - ended_at > self.VIDEO_KEEP_SECONDS):
            self.store.delete(video["path"])

    def stop(self):
        self._stop.set()
        if self.speaker is not None:
            close = getattr(self.speaker, 'close', None)
            stop_speech = close if close is not None else getattr(self.speaker, 'stop', None)
            if stop_speech is not None:
                stop_speech()
        self._thread.join(timeout=self.args.timeout + 3)
        if self.capture is not None:
            self.capture.release()
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            self._discard(*pending)
        self.store.cleanup()  # every uploaded video goes with the app

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
                self._apply_pending()
                if self.capture is None:
                    if self.video is not None or self.source_ended:
                        # An ended video waits for Replay, Back to camera or a new upload.
                        self._expire_ended_video()
                        self._stop.wait(0.05)
                    elif not self._try_open_camera():
                        self._stop.wait(0.05)
                    continue
                ok, frame, captured_at = self.capture.read()
                if not ok:
                    if self.capture.finished:
                        if self.video is not None:
                            self._video_finished()
                            continue
                        if self._is_camera:
                            self._camera_lost()  # Unplugged: keep the page up and retry.
                            continue
                        self.capture.release()
                        with self._lock:
                            self.capture = None
                            self.status.update(EMPTY_RESULT, error="Camera/source ended")
                            self.source_ended = True
                            self._latest_jpeg = self._latest_debug_jpeg = None
                            self._frame_captured_at = None
                        continue  # the page stays usable: uploads and Back to camera still work
                    if self.paused and self._last_frame is not None:
                        self._show_paused(self._last_frame)  # a paused video sends no frames
                    self._stop.wait(0.01)
                    continue
                position = getattr(self.capture, "position", 0.0)
                processing_started = time.monotonic()
                self._last_frame = frame
                with self._lock:
                    self._last_frame_at = time.monotonic()
                    paused = self.paused
                    mode = self.mode
                    generation = self._generation
                    view = self.view if self.video is not None or not self._is_camera else "alert"
                    if self._reset_policy:
                        self.policy.reset()
                        self._reset_policy = False
                if paused:
                    self._show_paused(frame)  # the camera stays open; nothing is inferred
                    continue
                error = ""
                batch = None
                result = None
                recognition = []
                spoken_at = None
                annotated = debug_annotated = frame

                # A local recording in Recognition view shows the analysed frame and its
                # boxes together. A slow inference still describes that exact frame.
                recorded_recognition = view == "recognition" and mode == "local"

                def check_fresh(what):
                    if not recorded_recognition and time.monotonic() - captured_at > self.args.max_frame_age:
                        raise StaleFrame(what)
                def cancelled():
                    with self._lock:
                        return (self._stop.is_set() or self.paused or
                                generation != self._generation or self._pending is not None)
                try:
                    check_fresh('Captured frame is stale; discarded')
                    with self._detect_lock:
                        check_fresh('Captured frame expired while waiting for inference; discarded')
                        if cancelled():
                            continue
                        batch = self.detectors[mode].predict(frame)
                except StaleFrame as exc:
                    error = str(exc)
                except Exception as exc:
                    if mode == "cloud":
                        self._fall_back(exc)
                        continue
                    error = str(exc)
                if cancelled():
                    continue
                if batch is not None and view == "recognition":
                    try:
                        # Recognition view (video only): every trained detection, no gate,
                        # no cooldown and no speech. The alert policy is not touched.
                        check_fresh('Inference result is stale; discarded')
                        hazards_seen = assess(batch.detections, (frame.shape[1], frame.shape[0]),
                                              self.settings['alert_gate'], self.settings['severity'])
                        recognition = media.recognition_dicts(hazards_seen)
                        annotated = debug_annotated = annotate_recognition(frame, hazards_seen)
                    except Exception as exc:
                        error = str(exc)
                elif batch is not None:
                    try:
                        # One decision per frame: banner, list, highlight and speech all
                        # come from these gated hazards (decision D6).
                        with self._detect_lock:
                            result = decide(frame, batch, self.policy, self.depth_estimator,
                                            check=lambda: check_fresh('Inference result is stale; discarded'))
                        recognition = media.recognition_dicts(result.hazards)
                        annotated, debug_annotated = annotate(frame, result)
                        if result.speak is not None and self._speak(result.speak, generation=generation):
                            spoken_at = time.time()
                    except Exception as exc:
                        # Never retain prior boxes/alerts after an inference failure.
                        error = str(exc)
                if error:
                    batch = result = None
                    recognition = []
                    annotated = debug_annotated = frame

                encode_started = time.monotonic()
                ok_jpeg, buffer = cv2.imencode('.jpg', limit_size(annotated, media.MAX_DISPLAY_SIDE),
                                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                if debug_annotated is annotated:
                    ok_debug, debug_buffer = ok_jpeg, buffer
                else:
                    ok_debug, debug_buffer = cv2.imencode('.jpg', limit_size(debug_annotated, media.MAX_DISPLAY_SIDE),
                                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                now = time.monotonic()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                hazards = sorted(result.hazards, key=lambda h: h.index) if result else []
                with self._lock:
                    if self._stop.is_set() or self.paused or generation != self._generation:
                        continue
                    self._hold_until = None
                    if ok_jpeg and ok_debug:
                        self._latest_jpeg = buffer.tobytes()
                        self._latest_debug_jpeg = debug_buffer.tobytes()
                        self._frame_captured_at = None if recorded_recognition else captured_at
                    self.status["fps"] = round(fps, 1)
                    self.status["alert"] = result.banner if result else ""
                    self.status["alerts"] = result.alerts if result else []
                    self.status["selected"] = _selected_dict(result.selected) if result else None
                    if spoken_at is not None:
                        self.status["spoken_at"] = spoken_at
                    self.status["error"] = error
                    self.status["latency_ms"] = batch.latency_ms if batch else None
                    self.status["round_trip_ms"] = batch.round_trip_ms if batch else None
                    self.status['pipeline_latency_ms'] = 1000 * (now-captured_at) if batch else None
                    self.status['processing_ms'] = 1000 * (now-processing_started) if batch else None
                    self.status['preview_encode_ms'] = 1000 * (now-encode_started) if batch else None
                    self.status["detections"] = [self._detection_dict(hazard, frame.shape[0],
                        calibration_applicable=self._is_camera and self.video is None) for hazard in hazards]
                    self.status["recognition"] = recognition
                    if self.video is not None and batch is not None:
                        self.summary.add(position, recognition)
                    if self.video is not None:
                        self.video_position = position
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
        return render_template("index.html", classes=getattr(state, 'settings', {}).get('classes', CLASS_NAMES))

    @app.route("/video_feed")
    def video_feed():
        # ?debug=1 also draws ignored (far, side, low-confidence) boxes and sizes.
        debug = request.args.get("debug") == "1"

        def generate():
            previous = None
            while not state._stop.is_set():
                jpeg = state.get_jpeg(debug)
                if jpeg is not None and jpeg is not previous:
                    previous = jpeg
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

    # Multipart limit for a batch of 20 images of 10 MB; each file is checked on its own.
    app.config["MAX_CONTENT_LENGTH"] = MAX_BATCH_BYTES

    @app.errorhandler(413)
    def too_large(_):
        if request.path == "/source":
            return jsonify(error="The video is larger than 150 MB."), 413
        return jsonify(error=f"The upload is too large: at most {media.MAX_BATCH} images, "
                             "each up to 10 MB."), 413

    @app.get("/config")
    def config():
        """What the page needs to draw the legend and controls, from the config."""
        return jsonify(priority=priority_order(state.settings["severity"]), band_labels=BAND_LABELS,
                       speech_rate_min=RATE_MIN, speech_rate_max=RATE_MAX, speech_rate_default=RATE_DEFAULT,
                       shortcuts={"Space": "pause", "M": "mute", "C": "mode"},
                       max_upload_bytes=MAX_UPLOAD_BYTES, max_batch=media.MAX_BATCH,
                       classes=list(CLASS_NAMES), views=list(VIEWS),
                       max_video_bytes=media.MAX_VIDEO_BYTES, max_video_seconds=media.MAX_VIDEO_SECONDS,
                       video_types=list(media.VIDEO_SUFFIXES),
                       accuracy_note=getattr(state, "accuracy_note", None) or media.accuracy_note(load_config()))

    @app.get("/samples")
    def samples():
        return jsonify([{"id": key, "name": ("Object demo " + key.removeprefix("object-") if key.startswith("object-") else f"Sample {i}")}
                        for i, key in enumerate(state.samples, 1)])

    def read_image(file):
        """One uploaded file: its bytes up to the limit, so a huge file is not read whole."""
        data = file.stream.read(MAX_UPLOAD_BYTES + 1)
        return decode_upload(data)

    @app.post("/analyze")
    def analyze():
        """Multipart field `image` (one or more JPEG or PNG files) or form field
        `sample` (one or more ids from /samples). Optional: `view` (recognition or
        alert, default alert), `expected` (a class name), `compare=1`, `speak=1`,
        `debug=1`. Nothing is stored.

        One image keeps the original response (its fields at the top level and its
        error as the HTTP status). Every response also has `results`, one entry per
        image, so a bad file in a batch is a row with an error."""
        form = request.form
        view = form.get("view") or "alert"
        try:
            if view not in VIEWS:
                raise UploadError('view must be "recognition" or "alert"')
            try:
                expected = media.valid_expected(form.get("expected"))
            except ValueError as exc:
                raise UploadError(str(exc))
            files = request.files.getlist("image")
            sample_ids = form.getlist("sample")
            if not files and not any(sample_ids) and not form.getlist("demo") and not form.get("demo_class"):
                raise UploadError("Choose an image file or a sample.")
            if len(files) + len(sample_ids) > media.MAX_BATCH:
                raise UploadError(f"Upload at most {media.MAX_BATCH} images at once.")
        except UploadError as exc:
            return jsonify(error=str(exc)), exc.status
        items, statuses = [], []
        for number, file in enumerate(files, 1):
            name = media.display_name(file.filename, f"Image {number}")
            try:
                items.append((name, "upload", read_image(file), None))
                statuses.append(200)
            except UploadError as exc:
                items.append((name, "upload", None, str(exc)))
                statuses.append(exc.status)
        library = media.demo_library(state.demo_root)
        demo_ids = form.getlist("demo")
        demo_class = form.get("demo_class")
        if demo_class:
            if demo_class not in CLASS_NAMES:
                return jsonify(error="Unknown class."), 400
            demo_ids += [key for key, item in library["images"].items() if item["class"] == demo_class]
        if len(files) + len(sample_ids) + len(demo_ids) > media.MAX_BATCH:
            if demo_class and not files and not sample_ids:
                demo_ids = demo_ids[:media.MAX_BATCH]  # a large class folder: its first 20 images
            else:
                return jsonify(error=f"Upload at most {media.MAX_BATCH} images at once."), 400
        sample_keys = list(state.samples)
        for sample_id in sample_ids:
            path = state.samples.get(sample_id)
            name = ("Object demo " + sample_id.removeprefix("object-") if sample_id.startswith("object-") else f"Sample {sample_keys.index(sample_id) + 1}") if path is not None else "Unknown sample"
            try:
                if path is None:
                    raise UploadError("Unknown sample.", 404)
                items.append((name, name, decode_upload(path.read_bytes()), None))
                statuses.append(200)
            except UploadError as exc:
                items.append((name, name, None, str(exc)))
                statuses.append(exc.status)
        demo_classes = set()
        for demo_id in demo_ids:
            item = library["images"].get(demo_id)
            name = item["name"] if item is not None else "Unknown library image"
            try:
                if item is None:
                    raise UploadError("Unknown library image.", 404)
                demo_classes.add(item["class"])
                items.append((name, "Demo library", decode_upload(item["path"].read_bytes()), None))
                statuses.append(200)
            except UploadError as exc:
                items.append((name, "Demo library", None, str(exc)))
                statuses.append(exc.status)
        if expected is None and demo_ids and not files and not sample_ids and len(demo_classes) == 1:
            # Library images sit in their class folder, so that class is what they show.
            expected = demo_classes.pop()
        compare = form_flag(form, "compare")
        try:
            batch = state.analyze_batch(items, view=view, expected=expected, compare=compare,
                                        speak=form_flag(form, "speak"), debug=form_flag(form, "debug"))
        except ModeUnavailable as exc:
            return jsonify(error=str(exc)), 503
        if len(items) == 1 and not compare:
            entry = batch["results"][0]
            if "error" in entry:
                return jsonify(error=entry["error"], **batch), statuses[0]
            run = entry["runs"][0]
            if "error" in run:
                return jsonify(error=run["error"], **batch), 503
            return jsonify({**batch, **run, "name": entry["name"], "image": run["images"][view]})
        return jsonify(batch)

    @app.get("/demo")
    def demo():
        """The demo library: ids and names only. Built from the folder each time."""
        return jsonify(media.library_listing(media.demo_library(state.demo_root)))

    @app.get("/demo/thumb/<item_id>")
    def demo_thumbnail(item_id):
        item = media.demo_library(state.demo_root)["images"].get(item_id)
        if item is None:
            return jsonify(error="Unknown library image."), 404
        try:
            frame = decode_upload(item["path"].read_bytes())
        except UploadError as exc:
            return jsonify(error=str(exc)), exc.status
        ok, buffer = cv2.imencode(".jpg", limit_size(frame, media.THUMBNAIL_SIDE), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return jsonify(error="Could not make a thumbnail."), 500
        return Response(buffer.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/source")
    def source():
        """Play a video through the live pipeline instead of the camera: multipart
        field `video` (MP4, MOV or WebM, at most 150 MB and 120 s). The file gets a
        random private name and is deleted when it is replaced, on Back to camera,
        on an error and when the app stops."""
        request.max_content_length = media.MAX_VIDEO_BYTES + 1024 * 1024
        try:
            if "video" in request.files:
                return jsonify(state.upload_video(request.files["video"]))
            if request.form.get("demo"):
                item = media.demo_library(state.demo_root)["videos"].get(request.form["demo"])
                if item is None:
                    raise UploadError("Unknown library video.", 404)
                return jsonify(state.play_library_video(item["path"], item["name"]))
            raise UploadError("Choose a video file.")
        except UploadError as exc:
            return jsonify(error=str(exc)), exc.status

    @app.post("/source/control")
    def source_control():
        """JSON `action` (restart or camera) and/or `view` (recognition or alert)."""
        try:
            return jsonify(state.source_control(request.get_json(silent=True)))
        except ControlConflict as exc:
            return jsonify(error=str(exc), **state.source_state()), 409
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

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
    import signal
    import sys
    args = parse_args()
    # A plain kill (SIGTERM) runs the same shutdown as Ctrl+C, so uploaded videos are deleted.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
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
