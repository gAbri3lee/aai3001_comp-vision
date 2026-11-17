"""POST /api/detect: JPEG body in, shared-format detections out. Nothing is stored or logged."""
from http.server import BaseHTTPRequestHandler
import io
import json
import math
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image, UnidentifiedImageError

import _yolo_onnx

MAX_BODY = 4_400_000       # Vercel rejects request bodies over 4.5 MB
MAX_PIXELS = 16_000_000


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'image/jpeg':
            return self._send(415, {'error': 'Send a JPEG body with Content-Type: image/jpeg'})
        confidence = parse_qs(urlparse(self.path).query).get('confidence', [None])[0]
        if confidence is not None:
            try:
                confidence = float(confidence)
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError
            except ValueError:
                return self._send(400, {'error': 'confidence must be between 0 and 1'})
        try:
            length = int(self.headers.get('Content-Length', 0))
        except ValueError:
            length = 0
        if length <= 0:
            return self._send(400, {'error': 'Empty body'})
        if length > MAX_BODY:
            return self._send(413, {'error': 'Image too large; the page downsizes photos automatically'})
        data = self.rfile.read(length)
        try:
            with Image.open(io.BytesIO(data)) as source:
                if source.format != 'JPEG':
                    raise ValueError('Expected JPEG')
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError('Image too large')
                image = source.convert('RGB')
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return self._send(400, {'error': 'Invalid JPEG image'})
        try:
            self._send(200, _yolo_onnx.detect(image, confidence))
        except Exception:
            self._send(503, {'error': 'Detector inference failed'})

    def log_message(self, *args):  # no request logging: uploaded photos are private
        pass
