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
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while not self.stop.is_set():
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
