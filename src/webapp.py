"""VisionAid web UI: live capture, detection, alert gate, speech and page controls.

Local mode never sends a frame anywhere. When the app starts with --endpoint, a
cloud detector is also built and the page can switch to it (POST /mode). A cloud
timeout or error falls back to local, keeps the last good result for at most one
second, and stays on local until a successful "Retry cloud" (decision U2).
"""
import argparse
import copy
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, request
from config import add_detector_arguments, detector_settings
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
        self.speaker = None if args.no_tts else Speaker()
        self.policy = AlertPolicy(self.settings['alert_gate'], self.settings['severity'])
        # Optional distance estimate for the debug panel only (never spoken).
        self.calibration = distance.load_calibration()
        self._lock = threading.Lock()
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

        source = int(args.source) if str(args.source).isdigit() else args.source
        self.capture = LatestFrameCapture(source)
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
            if self._frame_is_stale() or self._hold_expired():
                return None
            return self._latest_debug_jpeg if debug else self._latest_jpeg

    def _health(self):
        """Caller holds the lock. Plain messages for problems while running."""
        health = []
        if self.fallback_active:
            health.append({"code": "cloud_unreachable", "severity": "warning",
                           "message": "Cloud unreachable, using local."})
        return health

    def get_status(self) -> dict:
        with self._lock:
            status = dict(self.status)
            status.update(self._mode_fields())
            if self._hold_expired():
                status.update(EMPTY_RESULT)
            elif self._frame_is_stale():
                status.update(EMPTY_RESULT, error="Frame is stale; waiting for fresh inference")
            status["health"] = self._health()
            return status

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
        self.capture.release()

    # Background loop ------------------------------------------------------

    def _loop(self):
        try:
            prev_time = time.monotonic()
            while not self._stop.is_set():
                ok, frame, captured_at = self.capture.read()
                if not ok:
                    if self.capture.finished:
                        with self._lock:
                            self.status.update(EMPTY_RESULT, error="Camera/source ended")
                            self._latest_jpeg = self._latest_debug_jpeg = None
                            self._frame_captured_at = None
                        break
                    self._stop.wait(0.01)
                    continue
                with self._lock:
                    mode = self.mode
                    if self._reset_policy:
                        self.policy.reset()
                        self._reset_policy = False
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
                        result = decide(frame, batch, self.policy, self.depth_estimator,
                                        check=lambda: check_fresh('Inference result is stale; discarded'))
                        annotated, debug_annotated = annotate(frame, result)
                        if result.speak is not None:
                            spoken_at = time.time()
                            if self.speaker is not None:
                                self.speaker.say(result.speak)
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
        finally:
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
