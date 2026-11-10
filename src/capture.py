"""Keep only the newest captured frame while inference is busy."""
import threading
import time
import cv2


class LatestFrameCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f'Could not open camera/source: {source}')
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.latest = None
        self.finished = False
        # A camera delivers frames in real time. A recorded file would otherwise be
        # decoded as fast as possible and end before inference sees most of it, so
        # file sources are paced at their own frame rate.
        fps = self.cap.get(cv2.CAP_PROP_FPS) if not isinstance(source, int) else 0
        self.frame_interval = 1.0 / fps if fps and fps > 0 else 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            next_frame_at = time.monotonic()
            while not self.stop.is_set():
                if self.frame_interval:
                    self.stop.wait(max(0.0, next_frame_at - time.monotonic()))
                    next_frame_at += self.frame_interval
                ok, frame = self.cap.read()
                if not ok:
                    break
                with self.lock:
                    self.latest = (frame, time.monotonic())
        finally:
            with self.lock:
                self.finished = True

    def read(self):
        with self.lock:
            if self.latest is None:
                return False, None, None
            frame, captured_at = self.latest
            self.latest = None
            return True, frame, captured_at

    def release(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.cap.release()
