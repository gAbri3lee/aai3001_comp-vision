"""General detections supplement trained objects without changing checkpoint IDs."""
import argparse
from types import SimpleNamespace
from unittest.mock import patch


from config import add_detector_arguments, detector_settings
from detection import Detection, DetectionBatch


def test_malformed_optional_output_preserves_trained_detections_and_recovers():
    from scene_detector import SceneDetector
    named = Detection(2, 'Stair', .8, (0, 30, 40, 60))
    trained = SimpleNamespace(predict=lambda frame: DetectionBatch((named,), 10, 'trained'))
    general = SimpleNamespace(predict=lambda *a, **k: [SimpleNamespace(boxes=None)])
    with patch('scene_detector.create_detector', return_value=trained):
        detector = SceneDetector({'device': 'cpu'}, general=general)
        assert detector.predict(SimpleNamespace(shape=(60, 100, 3))).detections == (named,)
        assert detector.general_error
        rows = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: []))
        general.predict = lambda *a, **k: [SimpleNamespace(boxes=SimpleNamespace(data=rows))]
        assert detector.predict(SimpleNamespace(shape=(60, 100, 3))).detections == (named,)
        assert detector.general_error is None


def test_scene_detector_keeps_named_boxes_and_adds_only_unmatched_objects():
    from scene_detector import SceneDetector
    parser = argparse.ArgumentParser()
    add_detector_arguments(parser)
    settings = detector_settings(parser.parse_args([]))
    named = Detection(1, 'People', .8, (10, 10, 40, 50))

    class Trained:
        def predict(self, frame):
            return DetectionBatch((named,), 10, 'yolov8n')

    class General:
        def predict(self, frame, **kwargs):
            # Same person, a separate chair, and invalid edge overshoot.
            rows = [[11, 10, 41, 50, .9, 0],
                             [60, 10, 90, 50, .85, 56],
                             [110, 10, 120, 50, .8, 56]]
            return [SimpleNamespace(boxes=SimpleNamespace(data=SimpleNamespace(
                cpu=lambda: SimpleNamespace(numpy=lambda: rows))))]

    with patch('scene_detector.create_detector', return_value=Trained()), \
         patch('scene_detector.load_general_model', return_value=General()):
        detector = SceneDetector(settings)
        batch = detector.predict(SimpleNamespace(shape=(60, 100, 3)))
    assert batch.detections == (Detection(-1, 'Object', .85, (60., 10., 90., 50.)), named)
    assert batch.latency_ms >= 10


def test_empty_general_predictions_do_not_invent_an_object():
    from scene_detector import SceneDetector

    class Trained:
        def predict(self, frame):
            return DetectionBatch((), 10, 'rfdetr-nano', 15)

    class General:
        def predict(self, frame, **kwargs):
            return [SimpleNamespace(boxes=SimpleNamespace(data=SimpleNamespace(
                cpu=lambda: SimpleNamespace(numpy=lambda: []))))]

    with patch('scene_detector.create_detector', return_value=Trained()), \
         patch('scene_detector.load_general_model', return_value=General()):
        batch = SceneDetector({'device': 'cpu'}).predict(SimpleNamespace(shape=(60, 100, 3)))
    assert batch.detections == ()
    assert batch.round_trip_ms == 15


def test_unmatched_coco_person_becomes_general_people():
    from scene_detector import SceneDetector
    named = Detection(2, 'Stair', .8, (0, 30, 40, 60))

    class Trained:
        def predict(self, frame):
            return DetectionBatch((named,), 10, 'yolov8n')

    class General:
        def predict(self, frame, **kwargs):
            # A person the trained model missed, and a chair (COCO 56).
            rows = [[50, 5, 70, 55, .7, 0], [75, 20, 95, 50, .6, 56]]
            return [SimpleNamespace(boxes=SimpleNamespace(data=SimpleNamespace(
                cpu=lambda: SimpleNamespace(numpy=lambda: rows))))]

    with patch('scene_detector.create_detector', return_value=Trained()), \
         patch('scene_detector.load_general_model', return_value=General()):
        batch = SceneDetector({'device': 'cpu'}).predict(SimpleNamespace(shape=(60, 100, 3)))
    assert batch.detections == (named, Detection(-1, 'People', .7, (50., 5., 70., 55.)),
                                Detection(-1, 'Object', .6, (75., 20., 95., 50.)))
    assert batch.detections[1].to_dict()['class_id'] == -1
