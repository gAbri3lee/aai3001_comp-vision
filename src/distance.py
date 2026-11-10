"""Optional pinhole distance ESTIMATE for the debug panel. Never spoken.

Only for classes with a known real height (People 1.7 m, Wet Floor Sign 0.65 m in
config). It needs a focal length measured with scripts/calibrate_distance.py and
stored in the git-ignored camera_calibration.json:

    distance_m = focal_px x real_height_m / box_height_px

A box cut off by the top or bottom of the frame is shorter than the object, which
makes the estimate too far, so no estimate is given for it.
"""
import json
import statistics

from config import resolve_path

CALIBRATION_FILE = 'camera_calibration.json'


def load_calibration(path=None):
    """Return {'focal_px', 'frame_height'} or None when the camera is not calibrated."""
    path = resolve_path(path or CALIBRATION_FILE)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    samples = data.get('samples', [])
    if not samples:
        return None
    return dict(focal_px=statistics.median(s['focal_px'] for s in samples),
                frame_height=samples[-1]['frame_height'], samples=len(samples))


def focal_from_measurement(box_height_px, distance_m, real_height_m):
    if box_height_px <= 0 or distance_m <= 0 or real_height_m <= 0:
        raise ValueError('Box height, distance and real height must be positive')
    return box_height_px * distance_m / real_height_m


def is_truncated(xyxy, frame_height, margin=2):
    return xyxy[1] <= margin or xyxy[3] >= frame_height - margin


def estimate(label, xyxy, frame_height, calibration, known_heights):
    """(distance_m or None, note). The note explains a missing estimate."""
    if calibration is None:
        return None, 'not calibrated'
    if label not in known_heights:
        return None, 'no known height for this class'
    if is_truncated(xyxy, frame_height):
        return None, 'box cut off by frame edge'
    # Focal length in pixels scales with the frame resolution.
    focal = calibration['focal_px'] * frame_height / calibration['frame_height']
    return focal * known_heights[label] / (xyxy[3] - xyxy[1]), 'estimate'
