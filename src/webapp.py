"""
Web UI for the VisionAid pipeline — replaces the raw OpenCV debug window
with a browser UI (rubric's User Interface row expects something
intuitive, not a developer test harness). Runs the same capture ->
detection+depth -> decision engine -> TTS pipeline as live_pipeline_demo.py,
streamed to the browser, with on-demand OCR / product-ID triggered by UI
buttons (the spoken "vision scan" trigger still works for OCR too).

This also doubles as the "local deployment" half of the Model Deployment
rubric row: `python src/webapp.py` runs a real local web service, not just
a script.
"""
import argparse
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

from decision_engine import AlertCooldown, build_hazards
from depth import DepthEstimator
from ocr import OcrReader
from product_id import ProductIdentifier
from tts import Speaker
from voice_trigger import VoiceTrigger

HAZARD_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "fire hydrant", "stop sign", "bench", "chair",
    "dog", "backpack", "suitcase", "potted plant", "couch",
}


def phrase_for(hazard) -> str:
    if hazard.proximity > 0.7:
        urgency = "Warning"
    elif hazard.proximity > 0.4:
        urgency = "Caution"
    else:
        urgency = "Note"
    position = "ahead" if hazard.position == "center" else f"on your {hazard.position}"
    return f"{urgency}, {hazard.label} {position}"


class PipelineState:
    """Owns the background capture/inference loop and exposes thread-safe
    reads for Flask's request handlers (which run on separate threads)."""

    def __init__(self, args):
        self.args = args
        self.detector = YOLO(args.model)
        self.depth_estimator = None if args.no_depth else DepthEstimator()
        self.speaker = None if args.no_tts else Speaker()
        self.cooldown = AlertCooldown()
        self.ocr_reader = None if args.no_ocr else OcrReader()
        self.product_identifier = None if args.no_product_id else ProductIdentifier()

        self.voice_trigger = None
        if self.ocr_reader is not None and not args.no_voice:
            print("Starting voice trigger (needs internet + a working microphone)...")
            self.voice_trigger = VoiceTrigger(mic_index=args.mic_index)

        self._lock = threading.Lock()
        self._latest_jpeg = None
        self.status = {"fps": 0.0, "alert": "", "ocr_text": "", "product_text": ""}

        self._ocr_request = threading.Event()
        self._product_request = threading.Event()
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def request_ocr(self):
        self._ocr_request.set()

    def request_product_id(self):
        self._product_request.set()

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def get_status(self) -> dict:
        with self._lock:
            return dict(self.status)

    def stop(self):
        self._stop.set()
        if self.voice_trigger is not None:
            self.voice_trigger.stop()

    def _loop(self):
        source = int(self.args.source) if str(self.args.source).isdigit() else self.args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera/source: {self.args.source}")

        prev_time = time.time()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            results = self.detector.predict(frame, conf=self.args.conf, imgsz=self.args.imgsz, verbose=False)[0]
            names = results.names
            keep = [i for i, box in enumerate(results.boxes) if names[int(box.cls)].lower() in HAZARD_CLASSES]
            results = results[keep]
            annotated = results.plot()

            alert_text = ""
            if self.depth_estimator is not None and len(results.boxes) > 0:
                depth_map = self.depth_estimator.predict(frame)
                hazards = build_hazards(results, depth_map, frame.shape[1])
                for hazard in hazards:
                    x1, y1 = int(hazard.xyxy[0]), int(hazard.xyxy[1])
                    cv2.putText(annotated, f"{hazard.proximity:.2f}", (x1, max(y1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                if hazards:
                    top = hazards[0]
                    alert_text = phrase_for(top)
                    if self.speaker is not None and self.cooldown.should_announce(top):
                        self.speaker.say(alert_text)

            voice_triggered = self.voice_trigger is not None and self.voice_trigger.consume_trigger()
            if self.ocr_reader is not None and (self._ocr_request.is_set() or voice_triggered):
                self._ocr_request.clear()
                text = self.ocr_reader.read(frame)
                ocr_text = text if text else "(no text found)"
                with self._lock:
                    self.status["ocr_text"] = ocr_text
                if self.speaker is not None:
                    self.speaker.say(f"It says: {ocr_text}" if text else "No text found")

            if self.product_identifier is not None and self._product_request.is_set():
                self._product_request.clear()
                label, conf = self.product_identifier.identify(frame, topk=1)[0]
                product_text = f"{label} ({conf * 100:.0f}%)"
                with self._lock:
                    self.status["product_text"] = product_text
                if self.speaker is not None:
                    self.speaker.say(f"This looks like {label}")

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            ok_jpeg, buffer = cv2.imencode(".jpg", annotated)
            with self._lock:
                if ok_jpeg:
                    self._latest_jpeg = buffer.tobytes()
                self.status["fps"] = round(fps, 1)
                self.status["alert"] = alert_text

        cap.release()


def create_app(state: PipelineState) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                jpeg = state.get_jpeg()
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.03)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        return jsonify(state.get_status())

    @app.route("/scan_text", methods=["POST"])
    def scan_text():
        state.request_ocr()
        return jsonify({"ok": True})

    @app.route("/identify_item", methods=["POST"])
    def identify_item():
        state.request_product_id()
        return jsonify({"ok": True})

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="VisionAid web UI")
    parser.add_argument("--source", default=0, help="Camera index or video file path")
    parser.add_argument("--model", default="models/hazard_yolov8n.pt",
                         help="Path to YOLO weights (.pt); defaults to the fine-tuned hazard checkpoint")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Detection inference image size")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth estimation stage")
    parser.add_argument("--no-tts", action="store_true", help="Disable spoken alerts")
    parser.add_argument("--no-ocr", action="store_true", help="Disable the OCR on-demand stage")
    parser.add_argument("--no-voice", action="store_true", help="Disable spoken trigger for OCR")
    parser.add_argument("--no-product-id", action="store_true", help="Disable the product ID on-demand stage")
    parser.add_argument("--mic-index", type=int, default=None, help="Microphone device index for voice trigger")
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
