"""
Live webcam test of the hazard-detection stage using a pretrained YOLO
checkpoint as a placeholder. Swap --model to your own fine-tuned best.pt
once it exists; the rest of the pipeline is unaffected.
"""
import argparse
import time

import cv2
from ultralytics import YOLO

# COCO classes loosely relevant to walking-path hazards, used to filter the
# generic pretrained model's output down to something resembling the real
# hazard set. Not needed once a hazard-specific fine-tuned model is used.
HAZARD_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "fire hydrant", "stop sign", "bench", "chair",
    "dog", "backpack", "suitcase", "potted plant", "couch",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam hazard-detection test")
    parser.add_argument("--source", default=0, help="Camera index or video file path")
    parser.add_argument("--model", default="models/hazard_yolov8n.pt",
                         help="Path to YOLO weights (.pt); defaults to the fine-tuned hazard checkpoint")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--hazard-only", action="store_true",
                         help="Filter detections to the walking-hazard class subset")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/source: {args.source}")

    prev_time = time.time()
    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame, stopping.")
            break

        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        if args.hazard_only:
            names = results.names
            keep = [i for i, box in enumerate(results.boxes)
                    if names[int(box.cls)].lower() in HAZARD_CLASSES]
            results = results[keep]

        annotated = results.plot()

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("VisionAid - Hazard Detection Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
