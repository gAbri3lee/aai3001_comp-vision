import tempfile
import unittest
from pathlib import Path
from PIL import Image
from evaluation import load_ground_truth, score_detections


class EvaluationTests(unittest.TestCase):
    def test_legacy_five_class_labels_drop_pillar_and_remap_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'labels').mkdir()
            Image.new('RGB', (100, 100)).save(root / 'images' / 'mixed.jpg')
            # Original v4 ids: 2 Pillar, 3 Stair, 4 Wet Floor Sign.
            (root / 'labels/mixed.txt').write_text('2 0.2 0.2 0.1 0.1\n3 0.5 0.5 0.2 0.2\n4 0.8 0.8 0.1 0.1\n')
            active = ['Escalator', 'People', 'Stair', 'Wet Floor Sign']
            legacy = ['Escalator', 'People', 'Pillar', 'Stair', 'Wet Floor Sign']
            gt, paths = load_ground_truth(root, active, legacy)
            self.assertEqual([c['name'] for c in gt['categories']], active)
            self.assertEqual(sorted(a['category_id'] for a in gt['annotations']), [3, 4])   # Stair, Wet Floor Sign
            self.assertEqual(gt['excluded_label_boxes'], {'Pillar': 1})
            outputs = {'mixed.jpg': {'detections': [
                {'class_id': 2, 'confidence': .9, 'box_xyxy': [40, 40, 60, 60]},
                {'class_id': 3, 'confidence': .9, 'box_xyxy': [75, 75, 85, 85]}]}}
            metrics = score_detections(gt, paths, outputs, active, .35, [])
            self.assertAlmostEqual(metrics['per_class']['Stair']['ap50'], 1.)
            self.assertAlmostEqual(metrics['per_class']['Stair']['ap50_95'], 1.)
            self.assertAlmostEqual(metrics['per_class']['Wet Floor Sign']['ap50'], 1.)
            self.assertIsNone(metrics['per_class']['Escalator']['ap50'])
            # Without the legacy list, five-class ids are refused instead of silently reinterpreted.
            with self.assertRaises(ValueError):
                load_ground_truth(root, active)

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
            self.assertEqual({key:metrics['per_class']['Pillar'][key] for key in ('tp','fp','fn')},
                             {'tp':1,'fp':1,'fn':0})
            self.assertAlmostEqual(metrics['per_class']['Pillar']['ap50'], 1.)
            empty = {p.name: {'detections': []} for p in paths}
            metrics = score_detections(gt, paths, empty, names, .35, [])
            self.assertEqual(metrics['recall'], 0.)
            self.assertEqual(metrics['map50'], 0.)
