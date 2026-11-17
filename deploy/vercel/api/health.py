"""GET /api/health: readiness and the model's class order (same shape as the RF-DETR service)."""
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _yolo_onnx


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            config = _yolo_onnx.load_config()
            _yolo_onnx.get_session()
            status, payload = 200, {'status': 'ready', 'model': 'yolov8n-onnx', 'classes': config['classes'],
                                    'input_size': config['input_size'],
                                    'class_thresholds': config['class_thresholds']}
        except Exception:
            status, payload = 503, {'status': 'unavailable'}
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass
