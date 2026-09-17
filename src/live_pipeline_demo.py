"""
Full software pipeline test on a live webcam feed:
capture -> detection + depth -> rule-based decision engine -> TTS output.

This stands in for the physical glasses pipeline described in CLAUDE.md.
Swap --model to a fine-tuned checkpoint once the real hazard dataset
exists; everything downstream (depth, decision engine, TTS) is unaffected.
"""
import argparse
import time

import cv2
from ultralytics import YOLO

from decision_engine import AlertCooldown, build_hazards
from depth import DepthEstimator
from ocr import OcrReader
from tts import Speaker
from voice_trigger import VoiceTrigger

HAZARD_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "fire hydrant", "stop sign", "bench", "chair",
    "dog", "backpack", "suitcase", "potted plant", "couch",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Full hazard-detection pipeline test")
    parser.add_argument("--source", default=0, help="Camera index or video file path")
    parser.add_argument("--model", default="models/hazard_yolov8n.pt",
                         help="Path to YOLO weights (.pt); defaults to the fine-tuned hazard checkpoint")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Detection inference image size")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth estimation stage")
    parser.add_argument("--no-tts", action="store_true", help="Disable spoken alerts")
    parser.add_argument("--no-ocr", action="store_true", help="Disable the OCR on-demand stage entirely")
    parser.add_argument("--no-voice", action="store_true",
                         help="Disable spoken trigger for OCR (push-to-talk 'o' key still works)")
    parser.add_argument("--mic-index", type=int, default=None, help="Microphone device index for voice trigger")
    return parser.parse_args()


def phrase_for(hazard) -> str:
    if hazard.proximity > 0.7:
        urgency = "Warning"
    elif hazard.proximity > 0.4:
        urgency = "Caution"
    else:
        urgency = "Note"
    position = "ahead" if hazard.position == "center" else f"on your {hazard.position}"
    return f"{urgency}, {hazard.label} {position}"


def main():
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    detector = YOLO(args.model)
    depth_estimator = None if args.no_depth else DepthEstimator()
    speaker = None if args.no_tts else Speaker()
    cooldown = AlertCooldown()

    ocr_reader = None if args.no_ocr else OcrReader()
    voice_trigger = None
    if not args.no_ocr and not args.no_voice:
        print("Starting voice trigger (needs internet + a working microphone)...")
        voice_trigger = VoiceTrigger(mic_index=args.mic_index)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/source: {args.source}")

    prev_time = time.time()
    last_ocr_text = ""
    last_ocr_time = 0.0
    print("Press 'q' to quit. Press 'o' or say 'vision scan' to read text.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame, stopping.")
            break

        results = detector.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        names = results.names
        keep = [i for i, box in enumerate(results.boxes) if names[int(box.cls)].lower() in HAZARD_CLASSES]
        results = results[keep]

        annotated = results.plot()

        if depth_estimator is not None and len(results.boxes) > 0:
            depth_map = depth_estimator.predict(frame)
            hazards = build_hazards(results, depth_map, frame.shape[1])

            for hazard in hazards:
                x1, y1 = int(hazard.xyxy[0]), int(hazard.xyxy[1])
                cv2.putText(annotated, f"{hazard.proximity:.2f}", (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            if hazards:
                top = hazards[0]
                message = phrase_for(top)
                if speaker is not None and cooldown.should_announce(top):
                    speaker.say(message)
                cv2.putText(annotated, f"Priority: {message}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if last_ocr_text and now - last_ocr_time < 6.0:
            cv2.putText(annotated, f"Read: {last_ocr_text[:60]}", (10, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.imshow("VisionAid - Full Pipeline Test", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        ocr_requested = key == ord("o") or (voice_trigger is not None and voice_trigger.consume_trigger())
        if ocr_requested and ocr_reader is not None:
            print("Running OCR on current frame...")
            text = ocr_reader.read(frame)
            last_ocr_text = text if text else "(no text found)"
            last_ocr_time = time.time()
            print(f"OCR result: {last_ocr_text}")
            if speaker is not None:
                speaker.say(f"It says: {last_ocr_text}" if text else "No text found")

    if voice_trigger is not None:
        voice_trigger.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
