"""Measure the camera focal length (pixels) from one object at a tape-measured distance.

Place a person (1.7 m by default) or a Wet Floor Sign (0.65 m) at a measured
distance, fully in frame, then either capture from the camera or pass a photo:

    python scripts/calibrate_distance.py --camera 0 --class People --distance 3.0
    python scripts/calibrate_distance.py --image shot.jpg --class "Wet Floor Sign" --distance 2.0

focal_px = box_height_px x distance_m / real_height_m. Each run adds a sample to the
git-ignored camera_calibration.json; the app uses the median. The app then shows a
distance ESTIMATE in its debug panel for People and Wet Floor Sign only. It is never
spoken. Use the same camera and resolution as the app.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import add_detector_arguments, detector_settings, load_config, resolve_path
from detectors import create_detector
from distance import CALIBRATION_FILE, focal_from_measurement, is_truncated, load_calibration


def grab_frame(camera):
    import cv2
    cap = cv2.VideoCapture(camera)
    try:
        if not cap.isOpened():
            raise SystemExit(f'Could not open camera {camera}')
        for _ in range(10):  # let exposure settle
            ok, frame = cap.read()
        if not ok:
            raise SystemExit('Camera returned no frame')
        return frame
    finally:
        cap.release()


def main():
    import cv2
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_detector_arguments(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--camera', type=int)
    parser.add_argument('--class', dest='label', required=True, choices=['People', 'Wet Floor Sign'])
    parser.add_argument('--distance', type=float, required=True, help='Tape-measured distance in metres')
    parser.add_argument('--real-height', type=float, default=None, help='Override the known real height in metres')
    parser.add_argument('--box-height-px', type=float, default=None,
                        help='Use this measured box height instead of running the detector')
    parser.add_argument('--output', default=CALIBRATION_FILE)
    args = parser.parse_args()
    known = load_config(args.config)['alert_gate']['known_heights_m']
    real_height = known[args.label] if args.real_height is None else args.real_height

    frame = cv2.imread(str(args.image)) if args.image else grab_frame(args.camera)
    if frame is None:
        raise SystemExit(f'Could not read {args.image}')
    frame_height = frame.shape[0]
    if args.box_height_px is None:
        detections = [d for d in create_detector(detector_settings(args)).predict(frame).detections
                      if d.label == args.label]
        if not detections:
            raise SystemExit(f'No {args.label} detected; adjust the scene or pass --box-height-px')
        box = max(detections, key=lambda d: d.box_xyxy[3] - d.box_xyxy[1]).box_xyxy
        if is_truncated(box, frame_height):
            raise SystemExit('The box touches the top or bottom of the frame, so it is cut off. '
                             'Step back until the whole object is visible.')
        box_height = box[3] - box[1]
    else:
        box_height = args.box_height_px

    if box_height > frame_height:
        raise SystemExit('Box height cannot exceed the frame height.')

    focal = focal_from_measurement(box_height, args.distance, real_height)
    path = resolve_path(args.output)
    if path.exists() and load_calibration(path) is None:
        raise SystemExit('Existing calibration is invalid. Choose a new output path or repair it first.')
    data = json.loads(path.read_text()) if path.exists() else {'samples': []}
    data['samples'].append(dict(focal_px=focal, frame_height=frame_height, frame_width=frame.shape[1],
                                label=args.label, distance_m=args.distance, real_height_m=real_height,
                                box_height_px=box_height, recorded=time.strftime('%Y-%m-%d %H:%M:%S')))
    path.write_text(json.dumps(data, indent=2) + '\n')
    calibration = load_calibration(path)
    print(f'Box height {box_height:.1f} px of {frame_height} px at {args.distance} m: focal {focal:.0f} px. '
          f'Median over {calibration["samples"]} sample(s): {calibration["focal_px"]:.0f} px. Saved {path}.')


if __name__ == '__main__':
    main()
