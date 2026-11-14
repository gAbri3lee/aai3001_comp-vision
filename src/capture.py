"""Keep only the newest captured frame while inference is busy."""
import threading
import time
import cv2


def limit_size(frame, max_side):
    """Downscale so the longest side is at most max_side. Box sizes are fractions of
    the frame, so the alert gate behaves the same on the smaller frame."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if not max_side or longest <= max_side:
        return frame
    scale = max_side / longest
    return cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA)


class LatestFrameCapture:
    def __init__(self, source, max_side=None):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f'Could not open camera/source: {source}')
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.latest = None
        self.finished = False
        self.max_side = max_side
        self.position = 0.0  # seconds into a file source, for the newest frame
        self._running = threading.Event()
        self._running.set()
        # A camera delivers frames in real time. A recorded file would otherwise be
        # decoded as fast as possible and end before inference sees most of it, so
        # file sources are paced at their own frame rate. Frames that inference is too
        # slow for are replaced by newer ones, never queued, so playback is never slowed
        # down or sped up.
        fps = self.cap.get(cv2.CAP_PROP_FPS) if not isinstance(source, int) else 0
        self.frame_interval = 1.0 / fps if fps and fps > 0 else 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            next_frame_at = time.monotonic()
            while not self.stop.is_set():
                if not self._running.is_set():
                    # Paused: the file position stays put and the clock restarts on resume.
                    self._running.wait(0.05)
                    next_frame_at = time.monotonic()
                    continue
                if self.frame_interval:
                    self.stop.wait(max(0.0, next_frame_at - time.monotonic()))
                    next_frame_at += self.frame_interval
                if self.stop.is_set():
                    break
                if not self._running.is_set():
                    continue
                ok, frame = self.cap.read()
                if not ok:
                    break
                position = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if self.frame_interval else 0.0
                frame = limit_size(frame, self.max_side)
                with self.lock:
                    self.latest = (frame, time.monotonic())
                    self.position = position or 0.0
        finally:
            with self.lock:
                self.finished = True

    def pause(self):
        self._running.clear()

    def resume(self):
        self._running.set()

    def read(self):
        with self.lock:
            if self.latest is None:
                return False, None, None
            frame, captured_at = self.latest
            self.latest = None
            return True, frame, captured_at

    def release(self):
        self.stop.set()
        self._running.set()
        self.thread.join(timeout=2)
        self.cap.release()
