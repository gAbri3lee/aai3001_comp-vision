"""
Monocular depth estimation stage (MiDaS_small), experimental and off in the
first-test commands. It is the software-only stand-in for a physical ToF sensor:
no hardware exists, so this is the sole depth signal when enabled. It is slow on a
laptop CPU (about 0.19 FPS for the full pipeline in Gabriel's measurement).

Output is a per-frame RELATIVE proximity map, normalized to [0, 1] where
higher = closer. It is NOT calibrated metric distance: MiDaS has no notion
of scale from a single monocular frame, so proximity values are only
meaningful for comparing hazards *within the same frame*, which is exactly
what the ranking in decision_engine.py needs.
"""
import time

import cv2
import numpy as np
import torch


class DepthEstimator:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)
        prediction = self.model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = prediction.cpu().numpy()
        lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
        return np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)



class ThrottledDepth:
    """Live-video wrapper: run the depth model on every `every`-th call and reuse its map in between.

    MiDaS costs about 100 ms per frame on a laptop CPU, several times the detectors. Relative depth changes
    slowly between neighbouring frames, so the current frame's boxes are ranked against a map at most
    `every - 1` depth calls old. The map is recomputed early when it is older than `max_age` seconds, when
    the frame size changes, or after reset() (the app calls it on a source change). Still images must use
    the estimator directly. every=1 recomputes on every call, the original behaviour.
    """

    def __init__(self, estimator, every=3, max_age=1.0, clock=time.monotonic):
        if type(every) is not int or every < 1:
            raise ValueError('Depth interval must be a positive integer')
        if not max_age > 0:
            raise ValueError('Depth map maximum age must be positive')
        self.estimator, self.every, self.max_age, self.clock = estimator, every, max_age, clock
        self.computed = self.reused = 0
        self.reset()

    def reset(self):
        self._map, self._at, self._reuses = None, None, 0

    def predict(self, frame_bgr):
        now = self.clock()
        stale = (self._map is None or self._map.shape != frame_bgr.shape[:2] or now - self._at > self.max_age
                 or self._reuses >= self.every - 1)
        if not stale:
            self._reuses += 1
            self.reused += 1
            return self._map
        self._map, self._at, self._reuses = self.estimator.predict(frame_bgr), now, 0
        self.computed += 1
        return self._map


def box_proximity(depth_map: np.ndarray, xyxy) -> float:
    """Median proximity (0-1, higher = closer) inside a detection box."""
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, depth_map.shape[1]), min(y2, depth_map.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.median(depth_map[y1:y2, x1:x2]))
