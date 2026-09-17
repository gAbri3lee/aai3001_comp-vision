"""
On-demand OCR stage (CLAUDE.md pipeline stage 2, secondary track). Runs
only when triggered by a voice command or push-to-talk key, not every
frame like the continuous detection/depth stages.
"""
import sys
import warnings

# EasyOCR's first-run model download writes a Unicode progress bar to
# stdout, which crashes on Windows consoles using a legacy (non-UTF-8)
# codepage. Force UTF-8 before importing it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# Harmless CPU-only noise from EasyOCR's internal DataLoader/quantization
# calls; silence it so real errors aren't lost in the scroll.
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*quantize_per_tensor.*")

import cv2
import easyocr

# Scoped for short sign/label text (per CLAUDE.md), not dense documents — a
# webcam at typical distance doesn't have the resolution/steadiness for a
# full page, and low-confidence reads there are usually gibberish rather
# than useful text, hence the confidence filter below.
#
# Stylized brand/logo text (e.g. packaging) is a genuinely hard case for
# general OCR even with tuning here — italic/custom fonts and foil glare
# push confidence down and distort character shapes. UPSCALE_FACTOR and a
# lower MIN_CONFIDENCE help recover small or uncertain-but-real reads, at
# the cost of occasionally letting more noise through.
MIN_CONFIDENCE = 0.25
UPSCALE_FACTOR = 1.6


class OcrReader:
    def __init__(self, languages=("en",)):
        self._reader = easyocr.Reader(list(languages), gpu=False)

    def read(self, frame_bgr) -> str:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
                           interpolation=cv2.INTER_CUBIC)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        results = self._reader.readtext(gray, detail=1)
        kept = [(bbox, text.strip()) for bbox, text, conf in results
                if conf >= MIN_CONFIDENCE and text.strip()]
        # Approximate reading order: top-to-bottom (bucketed so a line's
        # boxes stay together), then left-to-right within a line.
        kept.sort(key=lambda item: (round(item[0][0][1] / 20), item[0][0][0]))
        return " ".join(text for _, text in kept)
