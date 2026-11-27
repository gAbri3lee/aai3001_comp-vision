"""Boundary tests catch class loss, swapped IDs, stale results and wrong defaults."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from config import load_config, add_detector_arguments, detector_settings
from detection import Detection, DetectionBatch, validate_names
from decision_engine import build_hazards, phrase_for, AlertCooldown


class DetectionTests(unittest.TestCase):
    def test_all_five_classes_survive_payload_round_trip(self):
        names = ['Escalator', 'People', 'Pillar', 'Stair', 'Wet Floor Sign']
        payload = {'detections': [{'class_id': i, 'class': n, 'confidence': .8,
                                  'box_xyxy': [10, 20, 30, 40]} for i, n in enumerate(names)],
                   'latency_ms': 12, 'model': 'yolov8n'}
        batch = DetectionBatch.from_dict(payload, names, (100, 100))
        self.assertEqual(batch.to_dict(), payload)
        self.assertEqual(len(batch.detections), 5)

    def test_rejects_swapped_labels_and_nonfinite_values(self):
        names = load_config()['classes']
        validate_names(dict(enumerate(names)), names)
        with self.assertRaises(ValueError):
            validate_names(['People', 'Escalator', 'Pillar', 'Stair', 'Wet Floor Sign'], names)
        for field, value in [('class_id', -1), ('class', 'Person'),
                             ('confidence', float('nan')), ('confidence', 1.1),
                             ('box_xyxy', [0, 0, float('inf'), 10]),
                             ('box_xyxy', [20, 0, 10, 10]),
                             ('box_xyxy', [0, 0, 101, 10])]:
            item = {'class_id': 2, 'class': 'Stair', 'confidence': .8, 'box_xyxy': [0, 0, 10, 10]}
            item[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                DetectionBatch.from_dict({'detections': [item], 'latency_ms': 1, 'model': 'rfdetr-nano'}, names, (100, 100))

    def test_model_defaults_and_overrides_resolve_from_project(self):
        parser = argparse.ArgumentParser()
        add_detector_arguments(parser)
        local = detector_settings(parser.parse_args([]))
        self.assertEqual((local['confidence'], local['input_size'], local['nms_iou']), (.35, 640, .7))  # lowest class cut-off
        self.assertEqual(local['class_thresholds'], {'Escalator': .35, 'People': .35, 'Stair': .35, 'Wet Floor Sign': .35})
        self.assertTrue(local['weights'].is_absolute())
        self.assertEqual(local['weights'].name, 'yolov8n_v4_four_class_int8.onnx')
        cloud = detector_settings(parser.parse_args(['--backend', 'cloud', '--endpoint', 'http://localhost:9000']))
        self.assertEqual((cloud['confidence'], cloud['input_size']), (.4, 384))
        override = detector_settings(parser.parse_args(['--conf', '.6', '--imgsz', '320', '--model', 'other.pt']))
        self.assertEqual((override['confidence'], override['input_size']), (.6, 320))
        self.assertEqual(override['weights'].name, 'other.pt')

    def test_config_rejects_reordered_classes(self):
        config = load_config()
        config['classes'].reverse()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.json'
            p.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(p)

    def test_no_depth_still_ranks_all_classes_and_announces_location(self):
        names = load_config()['classes']
        detections = [Detection(i, n, .8, (0, 0, 20, 20)) for i, n in enumerate(names)]
        hazards = build_hazards(detections, None, 100)
        self.assertEqual([h.label for h in hazards], ['Stair', 'Escalator', 'Wet Floor Sign', 'People'])
        self.assertEqual(hazards[0].position, 'left')
        self.assertIsNone(hazards[0].proximity)
        self.assertIn('distance unavailable', phrase_for(hazards[0]).lower())
        self.assertEqual(build_hazards([], None, 100), [])
        cooldown = AlertCooldown()
        self.assertTrue(cooldown.should_announce(hazards[0]))
        self.assertFalse(cooldown.should_announce(hazards[0]))

    def test_depth_formula_and_three_position_buckets(self):
        import numpy as np
        detections = [Detection(2, 'Stair', .8, (x, 0, x + 10, 10)) for x in [0, 45, 90]]
        hazards = build_hazards(detections, np.full((10, 100), .8), 100)
        self.assertEqual([h.position for h in hazards], ['left', 'center', 'right'])
        self.assertAlmostEqual(hazards[0].priority, .9)
        self.assertTrue(phrase_for(hazards[1]).startswith('Warning'))

    def test_unidentified_objects_announce_their_relative_position(self):
        for x, expected in [(0, 'Object detected on your left'),
                            (45, 'Object detected ahead'),
                            (90, 'Object detected on your right')]:
            with self.subTest(position=expected):
                detection = Detection(-1, 'Object', .8, (x, 0, x + 10, 10))
                hazard = build_hazards([detection], None, 100)[0]
                self.assertEqual(phrase_for(hazard), expected)
                self.assertTrue(AlertCooldown().should_announce(hazard))

    def test_named_hazards_keep_priority_over_distant_generic_objects(self):
        detections = [Detection(-1, 'Object', .9, (0, 0, 10, 10)),
                      Detection(2, 'Stair', .8, (45, 0, 55, 10))]
        hazards = build_hazards(detections, None, 100)
        self.assertEqual([h.label for h in hazards], ['Stair', 'Object'])

    def test_generic_notice_keeps_its_wording_with_depth(self):
        import numpy as np
        detection = Detection(-1, 'Object', .9, (45, 0, 55, 10))
        hazard = build_hazards([detection], np.full((10, 100), .9), 100)[0]
        self.assertAlmostEqual(hazard.proximity, .9)
        self.assertEqual(phrase_for(hazard), 'Object detected ahead')
