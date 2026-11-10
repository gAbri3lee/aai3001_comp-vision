"""JPEG-only detector service. Run behind the hosting platform's HTTPS/access control."""
import io
import math
import os
import threading
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, request

from config import load_config, resolve_path


def create_app(detector=None, config=None):
    config = config or load_config(os.environ.get('VISIONAID_CONFIG'))
    if detector is None:
        from rfdetr_detector import RFDetrDetector
        detector = RFDetrDetector(config, resolve_path(os.environ.get('VISIONAID_WEIGHTS', config['cloud']['file'])),
                                  os.environ.get('VISIONAID_DEVICE', 'cpu'))
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
    lock = threading.Lock()

    @app.get('/health')
    def health():
        return jsonify(status='ready', model='rfdetr-nano', classes=config['classes'],
                       input_size=config['cloud']['input_size'], class_thresholds=config['cloud']['class_thresholds'])

    @app.post('/detect')
    def detect():
        if request.mimetype != 'image/jpeg':
            return jsonify(error='Send a JPEG body with Content-Type: image/jpeg'), 415
        confidence = request.args.get('confidence')
        if confidence is not None:
            try:
                confidence = float(confidence)
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError('Invalid confidence')
            except ValueError:
                return jsonify(error='confidence must be between 0 and 1'), 400
        try:
            with Image.open(io.BytesIO(request.get_data())) as source:
                if source.format != 'JPEG':
                    raise ValueError('Expected JPEG')
                if source.width * source.height > 16_000_000:
                    raise ValueError('Image too large')
                image = source.convert('RGB')
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error='Invalid JPEG image'), 400
        if not lock.acquire(blocking=False):
            return jsonify(error='Detector busy; submit a newer frame'), 503
        try:
            result = detector.predict(image) if confidence is None else detector.predict(image, confidence=confidence)
            return jsonify(result.to_dict())
        except Exception:
            app.logger.exception('Detector inference failed')
            return jsonify(error='Detector inference failed'), 503
        finally:
            lock.release()

    return app


if __name__ == '__main__':
    create_app().run(host='0.0.0.0', port=9000, threaded=True)
