"""The Vercel function's numpy-only ONNX detector must reproduce the golden detections."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / 'deploy' / 'vercel'
sys.path.insert(0, str(ROOT / 'src'))

try:
    import onnxruntime  # noqa: F401
    HAVE_RUNTIME = True
except ImportError:
    HAVE_RUNTIME = False
MODEL_READY = (API / 'model').is_dir() and any((API / 'model').glob('*.onnx')) and (API / 'config.json').is_file()


@unittest.skipUnless(HAVE_RUNTIME and MODEL_READY, 'run deploy/vercel/prepare.py and install onnxruntime')
class VercelDetectorTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(API))
        import yolo_onnx
        self.detector = yolo_onnx

    def tearDown(self):
        sys.path.remove(str(API))

    def test_matches_golden_outputs(self):
        from PIL import Image
        from golden import compare_outputs
        reference = json.loads((ROOT / 'golden/expected/local_yolov8n.json').read_text())
        outputs = {}
        for name in reference:
            with Image.open(ROOT / 'golden/images' / name) as image:
                outputs[name] = self.detector.detect(image.convert('RGB'))
        result = compare_outputs(reference, outputs)
        self.assertTrue(result['passed'], result)

    def test_boxes_stay_inside_non_square_image(self):
        from PIL import Image
        image = Image.new('RGB', (1000, 300), (120, 120, 120))
        for d in self.detector.detect(image, confidence=0.01)['detections']:
            x1, y1, x2, y2 = d['box_xyxy']
            self.assertTrue(0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 300)


if __name__ == '__main__':
    unittest.main()
