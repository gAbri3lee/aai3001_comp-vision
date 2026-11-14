"""Banner, highlight and speech come from the same gated hazards of the same frame."""
import argparse
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from config import add_detector_arguments, load_config
from decision_engine import AlertPolicy, phrase_for
from detection import Detection, DetectionBatch

W, H = 1280, 720
CONFIG = load_config()


def box(x_centre, height, width=.1):
    cx, w, h = x_centre * W, width * W, height * H
    return (cx - w / 2, H - h, cx + w / 2, H)


FAR_PERSON = Detection(1, 'People', .9, box(.5, .1))
NEAR_STAIR = Detection(3, 'Stair', .8, box(.5, .4))
MID_PERSON = Detection(1, 'People', .7, box(.45, .45))
SIDE_CHAIR = Detection(-1, 'Object', .9, box(.1, .8))


def policy():
    return AlertPolicy(CONFIG['alert_gate'], CONFIG['severity'], clock=lambda: 0.)


def test_spoken_phrase_belongs_to_the_gated_list_of_the_same_frame():
    p = policy()
    for t, detections in enumerate([(FAR_PERSON, NEAR_STAIR, MID_PERSON, SIDE_CHAIR),
                                    (FAR_PERSON, MID_PERSON), (NEAR_STAIR,), ()]):
        result = p.process(detections, (W, H), now=float(t * 5))
        gated = [h for h in result.hazards if h.gated]
        assert result.banner == (phrase_for(gated[0]) if gated else '')
        if result.speak is not None:
            assert result.speak in result.alerts
            assert result.selected in gated
        if result.selected is not None:
            assert result.selected.gated


def test_far_person_never_reaches_the_speaker():
    p = policy()
    for t in range(20):
        result = p.process((FAR_PERSON, SIDE_CHAIR), (W, H), now=float(t))
        assert result.speak is None and result.banner == '' and result.selected is None


def test_cooldown_lets_the_next_hazard_speak_while_the_banner_stays_on_top():
    p = policy()
    first = p.process((NEAR_STAIR, MID_PERSON), (W, H), now=0.)
    assert first.speak == 'Warning, Stair ahead' and first.banner == first.speak
    second = p.process((NEAR_STAIR, MID_PERSON), (W, H), now=0.5)
    assert second.banner == 'Warning, Stair ahead'
    assert second.speak == 'Caution, People ahead' and second.selected.label == 'People'
    third = p.process((NEAR_STAIR, MID_PERSON), (W, H), now=1.)
    assert third.speak is None and third.selected.label == 'People'  # still marked while in view
    gone = p.process((NEAR_STAIR,), (W, H), now=1.5)
    assert gone.selected is None


def _args():
    parser = argparse.ArgumentParser()
    add_detector_arguments(parser)
    args = parser.parse_args([])
    args.no_depth, args.no_tts, args.source = True, False, 0
    return args


class Capture:
    finished = False
    def __init__(self, source): pass
    def read(self):
        time.sleep(.01)
        return True, np.zeros((H, W, 3), dtype=np.uint8), time.monotonic()
    def release(self): pass


def test_webapp_status_speech_and_banner_agree():
    from webapp import PipelineState, create_app
    spoken = []

    class Predictor:
        def predict(self, frame):
            return DetectionBatch((FAR_PERSON, NEAR_STAIR, SIDE_CHAIR), 10, 'scene')

    with patch('webapp.create_detector', return_value=Predictor()), \
         patch('webapp.LatestFrameCapture', Capture), \
         patch.dict('sys.modules', {'tts': SimpleNamespace(Speaker=lambda: SimpleNamespace(say=spoken.append))}):
        state = PipelineState(_args())
        try:
            deadline = time.monotonic() + 2
            while not spoken and time.monotonic() < deadline:
                time.sleep(.01)
            time.sleep(.1)
            client = create_app(state).test_client()
            status = client.get('/status').json
        finally:
            state.stop()
    assert spoken == ['Warning, Stair ahead']  # far person and side chair never spoken
    assert status['alert'] == 'Warning, Stair ahead' and status['alerts'] == ['Warning, Stair ahead']
    assert status['selected']['phrase'] == 'Warning, Stair ahead'
    assert status['selected']['band'] == 'near' and status['selected']['position'] == 'center'
    assert isinstance(status['spoken_at'], float)
    assert status['gate_enabled'] is True
    by_class = {d['class']: d for d in status['detections']}
    assert [d['class'] for d in status['detections']] == ['People', 'Stair', 'Object']  # detector order
    assert by_class['People']['gated'] is False and by_class['People']['reason'] == 'far'
    assert by_class['Object']['reason'] == 'side' and by_class['Object']['source'] == 'general'
    assert by_class['Stair']['gated'] is True and by_class['Stair']['band'] == 'near'
    assert abs(by_class['Stair']['size'] - .4) < 1e-6
    for detection in status['detections']:
        assert {'class_id', 'class', 'confidence', 'box_xyxy', 'size', 'band', 'position', 'gated', 'source'} <= set(detection)


def test_page_has_spoken_marker_ignored_section_debug_view_and_distance_caveat():
    from webapp import create_app
    page = create_app(SimpleNamespace()).test_client().get('/').get_data(as_text=True)
    for text in ('Ignored detections', 'spoken', 'debug=1', 'alert gate',
                 'Distance is an estimate from box size, not measured metres.'):
        assert text in page
    assert '—' not in page and '–' not in page
