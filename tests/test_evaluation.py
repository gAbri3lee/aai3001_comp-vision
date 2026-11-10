import tempfile
import unittest
from pathlib import Path
from PIL import Image
from evaluation import load_ground_truth, score_detections


class EvaluationTests(unittest.TestCase):
    def test_background_without_label_counts_false_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'labels').mkdir()
            for name in ['pillar.jpg', 'background.jpg']:
                Image.new('RGB', (100, 100)).save(root / 'images' / name)
            (root / 'labels/pillar.txt').write_text('2 0.5 0.5 0.2 0.2\n')
            names = ['Escalator', 'People', 'Pillar', 'Stair', 'Wet Floor Sign']
            gt, paths = load_ground_truth(root, names)
            outputs = {'pillar.jpg': {'detections': [{'class_id': 2, 'confidence': .9, 'box_xyxy': [40, 40, 60, 60]}]},
                       'background.jpg': {'detections': [{'class_id': 2, 'confidence': .8, 'box_xyxy': [0, 0, 10, 10]}]}}
            metrics = score_detections(gt, paths, outputs, names, .35, ['Stair', 'Escalator', 'Wet Floor Sign'])
            self.assertEqual(metrics['images'], 2)
            self.assertEqual(metrics['precision'], .5)
            self.assertEqual(metrics['recall'], 1.)
            self.assertEqual(metrics['per_class']['Pillar']['boxes'], 1)
            self.assertAlmostEqual(metrics['per_class']['Pillar']['ap50'], 1.)
            empty = {p.name: {'detections': []} for p in paths}
            metrics = score_detections(gt, paths, empty, names, .35, [])
            self.assertEqual(metrics['recall'], 0.)
            self.assertEqual(metrics['map50'], 0.)
