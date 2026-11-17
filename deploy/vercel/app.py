"""VisionAid photo check: Flask app for Vercel (also runs locally with `python app.py`).

POST /api/detect  JPEG body in, shared-format detections out. Nothing is stored or logged.
GET  /api/health  readiness + class order.
GET  /            the static page in public/.
"""
import io
import math
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

import yolo_onnx

HERE = Path(__file__).resolve().parent
MAX_PIXELS = 16_000_000

app = Flask(__name__, static_folder=str(HERE / 'public'), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 4_400_000     # Vercel rejects request bodies over 4.5 MB


@app.after_request
def no_store(response):
    response.headers['Cache-Control'] = 'no-store' if request.path.startswith('/api/') else response.headers.get('Cache-Control', 'public, max-age=0')
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


@app.get('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.get('/api/health')
def health():
    try:
        config = yolo_onnx.load_config()
        yolo_onnx.get_session()
    except Exception:
        return jsonify(status='unavailable'), 503
    return jsonify(status='ready', model='yolov8n-onnx', classes=config['classes'],
                   input_size=config['input_size'], class_thresholds=config['class_thresholds'])


@app.post('/api/detect')
def detect():
    if request.mimetype != 'image/jpeg':
        return jsonify(error='Send a JPEG body with Content-Type: image/jpeg'), 415
    confidence = request.args.get('confidence')
    if confidence is not None:
        try:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError
        except ValueError:
            return jsonify(error='confidence must be between 0 and 1'), 400
    data = request.get_data()
    if not data:
        return jsonify(error='Empty body'), 400
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format != 'JPEG':
                raise ValueError('Expected JPEG')
            if source.width * source.height > MAX_PIXELS:
                raise ValueError('Image too large')
            image = source.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify(error='Invalid JPEG image'), 400
    try:
        return jsonify(yolo_onnx.detect(image, confidence))
    except Exception:
        return jsonify(error='Detector inference failed'), 503


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='Image too large; the page downsizes photos automatically'), 413


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    app.run(host='127.0.0.1', port=port)
