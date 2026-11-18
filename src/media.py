"""Uploaded media for the demo: batch limits, display names and honest scores.

Uploaded files are untrusted. A client file name is only ever shown on the page,
after cleaning, and is never used in a path.
"""
import math
import unicodedata

from config import CLASS_NAMES

# Decision V4: up to 20 images per batch, each a JPEG or PNG of at most 10 MB.
MAX_BATCH = 20
MAX_DISPLAY_SIDE = 1280   # annotated images sent back to the page are at most this big
NAME_LENGTH = 80


def display_name(filename, fallback):
    """A name to show on the page: the last path part of the client's file name,
    without control characters and cut to a sensible length. Never used as a path."""
    name = (filename or '').replace('\\', '/').rsplit('/', 1)[-1]
    name = ''.join(ch for ch in name if unicodedata.category(ch)[0] != 'C').strip()
    if len(name) > NAME_LENGTH:
        name = name[:NAME_LENGTH - 3] + '...'
    return name or fallback


def recognition_dicts(hazards):
    """Recognition view (decision V1): every trained detection that passed its class
    confidence threshold, with no distance, side or general-model filter. Highest
    confidence first."""
    trained = sorted((h for h in hazards if h.source == 'trained'), key=lambda h: -h.confidence)
    return [{'class_id': h.class_id, 'class': h.label, 'confidence': h.confidence,
             'box_xyxy': list(h.xyxy), 'position': h.position, 'size': h.size} for h in trained]


def expected_result(recognition, expected):
    """Decision V5: recognised when any detection of the expected class exists."""
    scores = [d['confidence'] for d in recognition if d['class'] == expected]
    return {'class': expected, 'recognised': bool(scores),
            'confidence': max(scores) if scores else None}


def batch_summary(results, backend, model, expected):
    """For example "Stair: 8 of 10 recognised, mean confidence 0.71". The mean is
    over the best expected-class confidence of each recognised image."""
    runs = [run for entry in results for run in entry.get('runs', ())
            if run.get('mode') == backend and 'error' not in run]
    scores = [run['expected']['confidence'] for run in runs if run['expected']['recognised']]
    mean = sum(scores) / len(scores) if scores else None
    skipped = sum(1 for entry in results if 'error' in entry) + sum(
        1 for entry in results for run in entry.get('runs', ()) if run.get('mode') == backend and 'error' in run)
    text = f'{expected}: {len(scores)} of {len(runs)} recognised'
    if mean is not None:
        text += f', mean confidence {mean:.2f}'
    return {'backend': backend, 'model': model, 'class': expected, 'recognised': len(scores),
            'total': len(runs), 'skipped': skipped,
            'mean_confidence': None if mean is None else round(mean, 2), 'text': text}


def _three_places(value):
    # Truncated, so a rounded figure never overstates the recorded result.
    return f'{math.floor(value * 1000) / 1000:.3f}'


def accuracy_note(config, subject='these images'):
    """The caption shown next to every score, from the recorded test results."""
    if config.get('benchmark_note'):
        return f"This is a check on {subject} only. {config['benchmark_note']}"
    test_images = config['dataset']['split_images']['test']
    return (f"This is a check on {subject} only. Tested accuracy on {test_images} test images: "
            f"local mAP50 {_three_places(config['local']['expected']['test_map50'])}, "
            f"cloud mAP50 {_three_places(config['cloud']['expected']['test_map50'])}.")


def valid_expected(value):
    if value in (None, ''):
        return None
    if value not in CLASS_NAMES:
        raise ValueError('expected must be one of: ' + ', '.join(CLASS_NAMES))
    return value


# Video upload (decision V3): MP4, MOV or WebM, at most 150 MB and 120 seconds.
VIDEO_SUFFIXES = (".mp4", ".mov", ".webm")
MAX_VIDEO_BYTES = 150 * 1024 * 1024
MAX_VIDEO_SECONDS = 120
MAX_VIDEO_SIDE = 7680        # larger frames are refused
INFERENCE_MAX_SIDE = 1920    # larger frames (for example 4K) are downscaled before inference
TEMP_PREFIX = "visionaid-video-"


class MediaError(ValueError):
    """A file the app will not use, with a message for the page and an HTTP status."""
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def video_suffix(name):
    """The type comes from the client's name only to pick one of our own suffixes."""
    suffix = '.' + name.rsplit('.', 1)[-1].lower() if '.' in name else ''
    if suffix not in VIDEO_SUFFIXES:
        raise MediaError("Upload an MP4, MOV or WebM video.", 415)
    return suffix


def looks_like_video(head: bytes):
    """MP4 and MOV start with an ISO media box; WebM with the EBML header."""
    return head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip") or head[:4] == b"\x1a\x45\xdf\xa3"


def probe_video(path):
    """Open the file with OpenCV and enforce the limits. Rotation metadata is
    applied by OpenCV (auto-orientation), so sizes are those of the upright picture."""
    import cv2
    with open(path, "rb") as file:
        if not looks_like_video(file.read(16)):
            raise MediaError("Upload an MP4, MOV or WebM video.", 415)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise MediaError("Could not read this video. The file may be damaged.")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise MediaError("Could not read this video. The file may be damaged.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or not math.isfinite(fps) or not 0 < fps <= 240:
            raise MediaError("Could not read the frame rate of this video.")
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not frames or not math.isfinite(frames) or frames <= 0:
            # Some containers do not record a frame count: count, but stop at the limit.
            frames, limit = 1, MAX_VIDEO_SECONDS * fps + 1
            while frames <= limit and capture.grab():
                frames += 1
        duration = frames / fps
        if duration > MAX_VIDEO_SECONDS:
            raise MediaError(f"The video is longer than {MAX_VIDEO_SECONDS} seconds ({duration:.0f} s).")
        height, width = frame.shape[:2]
        if max(width, height) > MAX_VIDEO_SIDE:
            raise MediaError("The video frames are too large.")
        return {"fps": round(fps, 3), "duration": round(duration, 2), "width": int(width), "height": int(height)}
    finally:
        capture.release()


class TempStore:
    """Private temporary folder for uploaded videos. Files get random names, are
    readable only by this user, and the folder is removed when the app stops. A
    folder left behind by a crashed run is removed at the next start."""

    def __init__(self):
        self.root = None

    def _folder(self):
        import os
        import tempfile
        from pathlib import Path
        if self.root is None:
            # mkdtemp creates the folder readable only by this user.
            self.root = Path(tempfile.mkdtemp(prefix=f"{TEMP_PREFIX}{os.getpid()}-"))
        return self.root

    def save(self, stream, suffix):
        """Copy an upload into a new private file, stopping at the size limit."""
        import os
        import tempfile
        from pathlib import Path
        handle, name = tempfile.mkstemp(suffix=suffix, dir=self._folder())
        path = Path(name)
        try:
            written = 0
            with os.fdopen(handle, "wb") as out:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_VIDEO_BYTES:
                        raise MediaError("The video is larger than 150 MB.", 413)
                    out.write(chunk)
            if written == 0:
                raise MediaError("The file is empty.")
            return path
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @staticmethod
    def delete(path):
        if path is not None:
            path.unlink(missing_ok=True)

    def cleanup(self):
        import shutil
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None

    @staticmethod
    def remove_stale():
        """Remove folders from earlier runs whose process has ended."""
        import os
        import shutil
        import tempfile
        from pathlib import Path
        for folder in Path(tempfile.gettempdir()).glob(TEMP_PREFIX + "*"):
            try:
                pid = int(folder.name[len(TEMP_PREFIX):].split("-", 1)[0])
                if (folder.is_symlink() or getattr(folder, 'is_junction', lambda: False)()
                        or not folder.is_dir()):
                    continue
                if hasattr(os, 'getuid') and folder.stat().st_uid != os.getuid():
                    continue
                if pid == os.getpid():
                    continue
                if os.name == 'nt':
                    # os.getuid is absent and os.kill is not a POSIX existence
                    # probe on Windows. Query only; never signal another process.
                    import ctypes
                    from ctypes import wintypes
                    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                    kernel.OpenProcess.restype = wintypes.HANDLE
                    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
                    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                    handle = kernel.OpenProcess(0x1000, False, pid)
                    if not handle:
                        if ctypes.get_last_error() in (87, 1168):
                            raise ProcessLookupError(pid)
                        continue  # access denied/unknown: leave the folder alone
                    try:
                        code = wintypes.DWORD()
                        if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259:
                            raise ProcessLookupError(pid)
                    finally:
                        kernel.CloseHandle(handle)
                else:
                    os.kill(pid, 0)  # still running: leave it alone
            except ProcessLookupError:
                shutil.rmtree(folder, ignore_errors=True)
            except (ValueError, OSError):
                continue


class ClassSummary:
    """Per-class record while a video plays (decision V8): first and last time a
    class was seen, in how many processed frames, and its highest confidence.
    Counts every trained detection, as in Recognition view, whatever the view."""

    def __init__(self):
        self.frames = 0
        self.classes = {}

    def add(self, position, recognition):
        self.frames += 1
        for label in {d["class"] for d in recognition}:
            best = max(d["confidence"] for d in recognition if d["class"] == label)
            entry = self.classes.setdefault(label, {"first_seen": position, "last_seen": position,
                                                    "frames": 0, "max_confidence": 0.0})
            entry["last_seen"] = position
            entry["frames"] += 1
            entry["max_confidence"] = max(entry["max_confidence"], best)

    def to_list(self):
        """All active classes, including those never seen."""
        rows = []
        for label in CLASS_NAMES:
            entry = self.classes.get(label)
            rows.append({"class": label, "seen": entry is not None,
                         "first_seen": None if entry is None else round(entry["first_seen"], 1),
                         "last_seen": None if entry is None else round(entry["last_seen"], 1),
                         "frames": 0 if entry is None else entry["frames"],
                         "max_confidence": None if entry is None else round(entry["max_confidence"], 2)})
        return rows


# Demo library (decision V7): demo/<Class>/ images and demo/videos/ videos, listed
# for one-click use. The allow-list is built from the folder on every request and
# clients choose by id only; a path from the client is never used.
DEMO_DIR = "demo"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
THUMBNAIL_SIDE = 240


def _library_id(prefix, text):
    import hashlib
    return f"{prefix}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def _inside(path, root):
    """A regular file inside the library folder: symbolic links that lead out of it
    and hidden files are left out."""
    try:
        return (not path.name.startswith('.') and path.is_file()
                and path.resolve().is_relative_to(root.resolve()))
    except OSError:
        return False


def demo_library(root):
    """{"images": {id: {class, name, path}}, "videos": {id: {name, path}}}."""
    images, videos = {}, {}
    if root.is_dir():
        for label in CLASS_NAMES:
            folder = root / label
            if folder.is_dir():
                for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
                    if path.suffix.lower() in IMAGE_SUFFIXES and _inside(path, root):
                        images[_library_id('img', f'{label}/{path.name}')] = {
                            'class': label, 'name': path.name, 'path': path}
        folder = root / 'videos'
        if folder.is_dir():
            for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
                if path.suffix.lower() in VIDEO_SUFFIXES and _inside(path, root):
                    videos[_library_id('vid', path.name)] = {'name': path.name, 'path': path}
    return {'images': images, 'videos': videos}


def library_listing(library):
    """What GET /demo returns: names and ids only, never paths."""
    classes = [{'class': label,
                'images': [{'id': key, 'name': item['name']}
                           for key, item in library['images'].items() if item['class'] == label]}
               for label in CLASS_NAMES]
    videos = [{'id': key, 'name': item['name']} for key, item in library['videos'].items()]
    return {'classes': classes, 'videos': videos,
            'empty': not library['images'] and not library['videos'],
            'help': ("The demo library is empty. Put images in demo/<class name>/ (for example "
                     "demo/Stair/) and videos in demo/videos/, then press Reload. See demo/README.md.")}
