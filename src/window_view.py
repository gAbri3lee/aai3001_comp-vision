"""Responsive OpenCV presentation, independent of the inference worker."""
import time
import cv2
import numpy as np


def run_window(state, title):
    try:
        while True:
            status = state.get_status()
            jpeg = state.get_jpeg()
            # Clear the previous annotated image while a request/source is stale.
            frame = (cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                     if jpeg is not None else np.zeros((480, 640, 3), dtype=np.uint8))
            text = status['error'] or status['alert'] or ('No detections' if jpeg else 'Waiting for camera')
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 200, 255), 2)
            cv2.putText(frame, f"{status['fps']:.1f} FPS | {status['model']}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 200, 255), 2)
            cv2.imshow(title, frame)
            key = cv2.waitKey(10) & 0xff
            if key == ord('q') or (state.capture.finished and not state._thread.is_alive()):
                break
            time.sleep(.01)
    finally:
        state.stop()
        cv2.destroyAllWindows()
