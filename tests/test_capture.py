"""Recorded files must play at their own frame rate, like a live camera."""
import time

import cv2
import numpy as np

from capture import LatestFrameCapture


def test_video_file_is_paced_at_its_frame_rate(tmp_path):
    path = tmp_path / 'clip.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (32, 32))
    for i in range(10):  # 0.5 s of video
        writer.write(np.full((32, 32, 3), i * 20, dtype=np.uint8))
    writer.release()

    start = time.monotonic()
    capture = LatestFrameCapture(str(path))
    try:
        assert abs(capture.frame_interval - 0.05) < 1e-6
        while not capture.finished and time.monotonic() - start < 5:
            time.sleep(0.01)
        elapsed = time.monotonic() - start
    finally:
        capture.release()
    # Unpaced decoding of ten tiny frames takes a few milliseconds.
    assert elapsed >= 0.4
