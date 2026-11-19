"""Native paths for dataset exports with long Windows video filenames."""
import os
from pathlib import Path


def native_path(path):
    path = Path(path).expanduser().resolve()
    value = str(path)
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = ('\\\\?\\UNC\\' + value[2:]) if value.startswith('\\\\') else '\\\\?\\' + value
        return Path(value)
    return path


def read_bgr(path):
    """Decode bytes instead of relying on OpenCV's platform filename support."""
    import cv2
    import numpy as np
    return cv2.imdecode(np.frombuffer(native_path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
