"""
Monocular depth estimation stage (MiDaS_small) — the software-only stand-in
for a physical ToF sensor (see CLAUDE.md: no hardware exists, so this is the
sole depth signal, not a fallback).

Output is a per-frame RELATIVE proximity map, normalized to [0, 1] where
higher = closer. It is NOT calibrated metric distance — MiDaS has no notion
of scale from a single monocular frame, so proximity values are only
meaningful for comparing hazards *within the same frame*, which is exactly
what the ranking in decision_engine.py needs.
"""
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


def box_proximity(depth_map: np.ndarray, xyxy) -> float:
    """Median proximity (0-1, higher = closer) inside a detection box."""
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, depth_map.shape[1]), min(y2, depth_map.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.median(depth_map[y1:y2, x1:x2]))
