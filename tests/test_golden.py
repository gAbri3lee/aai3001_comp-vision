import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from golden import compare_outputs


class GoldenTests(unittest.TestCase):
    def test_missing_and_unexpected_images_fail_even_for_background(self):
        ref = {'background.jpg': {'detections': []}}
        self.assertFalse(compare_outputs(ref, {})['passed'])
        self.assertFalse(compare_outputs(ref, {**ref, 'unexpected.jpg': {'detections': []}})['passed'])
        self.assertTrue(compare_outputs(ref, ref)['passed'])
        self.assertFalse(compare_outputs(ref, {'background.jpg': {'detections': [
            {'class_id': 2, 'confidence': .8, 'box_xyxy': [0, 0, 10, 10]}]}})['passed'])

    def test_pillar_requires_correct_id_and_box(self):
        ref = {'p.jpg': {'detections': [{'class_id': 2, 'confidence': .8, 'box_xyxy': [0, 0, 10, 10]}]}}
        self.assertTrue(compare_outputs(ref, ref)['passed'])
        got = {'p.jpg': {'detections': [{'class_id': 1, 'confidence': .8, 'box_xyxy': [0, 0, 10, 10]}]}}
        self.assertFalse(compare_outputs(ref, got)['passed'])

    def test_cli_returns_nonzero_on_failure(self):
        checker = Path(__file__).resolve().parents[1] / 'golden' / 'check_golden.py'
        with tempfile.TemporaryDirectory() as directory:
            reference, outputs = Path(directory) / 'reference.json', Path(directory) / 'outputs.json'
            reference.write_text(json.dumps({'background.jpg': {'detections': []}}))
            outputs.write_text('{}')
            result = subprocess.run([sys.executable, str(checker), str(reference), str(outputs)], capture_output=True)
            self.assertEqual(result.returncode, 1)
            outputs.write_text(reference.read_text())
            result = subprocess.run([sys.executable, str(checker), str(reference), str(outputs)], capture_output=True)
            self.assertEqual(result.returncode, 0)
