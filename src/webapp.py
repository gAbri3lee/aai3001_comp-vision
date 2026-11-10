"""Live object detection, relative depth and shared GUI/audio alerts."""
import argparse
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


def _selected_dict(hazard):
    if hazard is None:
        return None
    return {"class": hazard.label, "position": hazard.position, "band": hazard.band,
            "source": hazard.source, "phrase": phrase_for(hazard), "box_xyxy": list(hazard.xyxy)}

class PipelineState:
    """Owns the background capture/inference loop and exposes thread-safe
    reads for Flask's request handlers (which run on separate threads)."""

    def __init__(self, args):
        self.args = args
        self.settings = detector_settings(args)
        self.detector = create_detector(self.settings)
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
        gate = self.settings['alert_gate']
        self.status = {"fps": 0.0, "alert": "", "alerts": [], "selected": None, "spoken_at": None,
                       "model": self.settings["model"] + ' + general-yolov8n', "backend": args.backend, "error": "",
                       "latency_ms": None, "round_trip_ms": None, "detections": [],
                       "gate_enabled": gate['enabled'], "gate_calibrated": gate.get('calibrated', False),
                       "depth_enabled": not args.no_depth, "tts_enabled": not args.no_tts,
                       "distance_calibrated": self.calibration is not None}

        self._stop = threading.Event()

        source = int(args.source) if str(args.source).isdigit() else args.source
        self.capture = LatestFrameCapture(source)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _frame_is_stale(self):
        return (self._frame_captured_at is not None and
                time.monotonic() - self._frame_captured_at > self.args.max_frame_age)

    def get_jpeg(self, debug=False):
        with self._lock:
            if self._frame_is_stale():
                return None
            return self._latest_debug_jpeg if debug else self._latest_jpeg

    def get_status(self) -> dict:
        with self._lock:
            status = dict(self.status)
            if self._frame_is_stale():
                status.update(EMPTY_RESULT, error="Frame is stale; waiting for fresh inference")
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
                error = ""
                batch = None
                result = None
                spoken_at = None
                annotated = debug_annotated = frame
                try:
                    if time.monotonic() - captured_at > self.args.max_frame_age:
                        raise DetectorError('Captured frame is stale; discarded')
                    batch = self.detector.predict(frame)
                    depth_map = self.depth_estimator.predict(frame) if self.depth_estimator is not None and batch.detections else None
                    if time.monotonic() - captured_at > self.args.max_frame_age:
                        raise DetectorError('Inference result is stale; discarded')
                    # One decision per frame: banner, list, highlight and speech all
                    # come from these gated hazards (decision D6).
                    result = self.policy.process(batch.detections, (frame.shape[1], frame.shape[0]), depth_map)
                    annotated = annotate_hazards(frame, result.hazards, result.selected)
                    debug_annotated = annotate_hazards(frame, result.hazards, result.selected, debug=True)
                    if result.speak is not None:
                        spoken_at = time.time()
                        if self.speaker is not None:
                            self.speaker.say(result.speak)
                except Exception as exc:
                    # Never retain prior boxes/alerts after an inference failure.
                    error = str(exc)
                    batch = result = None
                    annotated = debug_annotated = frame

                now = time.monotonic()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                ok_jpeg, buffer = cv2.imencode(".jpg", annotated)
                ok_debug, debug_buffer = cv2.imencode(".jpg", debug_annotated)
                hazards = sorted(result.hazards, key=lambda h: h.index) if result else []
                with self._lock:
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
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        state.stop()


if __name__ == "__main__":
    main()
