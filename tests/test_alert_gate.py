"""Alert gate rules on synthetic detections (1280x720 frame, no weights needed)."""
import copy

import pytest

from config import CLASS_NAMES, load_config, validate_alert_gate
from decision_engine import AlertCooldown, assess, phrase_for
from detection import Detection

W, H = 1280, 720
GATE = load_config()['alert_gate']
SEVERITY = load_config()['severity']


def det(label, x_centre, height, confidence=0.8, general=False, width=0.1):
    """Box centred at x_centre (fraction of width) with the given height fraction."""
    class_id = -1 if general or label == 'Object' else CLASS_NAMES.index(label)
    cx, w, h = x_centre * W, width * W, height * H
    return Detection(class_id, label, confidence, (cx - w / 2, H - h, cx + w / 2, H))


def run(*detections, gate=GATE):
    return assess(detections, (W, H), gate, SEVERITY)


def spoken(*detections):
    return [phrase_for(h) for h in run(*detections) if h.gated]


def test_far_person_in_path_is_not_announced():
    [hazard] = run(det('People', .5, .10))
    assert not hazard.gated and hazard.reason == 'far' and hazard.band == 'far'


def test_near_person_in_path_is_a_warning():
    assert spoken(det('People', .5, .65)) == ['Warning, People ahead']


def test_side_person_needs_to_be_clearly_near():
    [hazard] = run(det('People', .15, .45))  # below 0.40 x 1.5 = 0.60
    assert not hazard.gated and hazard.reason == 'side'
    assert spoken(det('People', .15, .70)) == ['Warning, People on your left']


def test_wet_floor_sign_bands_follow_the_size_rule():
    # 0.16 / 0.28 = 0.57 is below the 0.6 mid-close boundary, so "Note".
    assert spoken(det('Wet Floor Sign', .5, .16)) == ['Note, Wet Floor Sign ahead']
    # 0.20 / 0.28 = 0.71 is mid-close by the B1 rule, so "Caution". (The hand-over
    # table listed "Note" for 0.20, which contradicts its own band rule.)
    assert spoken(det('Wet Floor Sign', .5, .20)) == ['Caution, Wet Floor Sign ahead']


def test_stair_in_path():
    assert spoken(det('Stair', .5, .40)) == ['Warning, Stair ahead']


def test_low_confidence_object_is_ignored():
    [hazard] = run(det('Object', .5, .40, confidence=.4))
    assert not hazard.gated and hazard.reason == 'low_confidence'


def test_side_object_is_ignored_even_when_large():
    [hazard] = run(det('Object', .15, .90, confidence=.9))
    assert not hazard.gated and hazard.reason == 'side'


def test_object_in_path_is_announced_below_every_trained_hazard():
    obj = det('Object', .5, .40, confidence=.6)
    assert spoken(obj) == ['Object detected ahead']
    for label in CLASS_NAMES:
        # The weakest trained hazard that still passes the gate.
        trained = det(label, .5, GATE['min_size'][label], confidence=.36)
        hazards = run(obj, trained)
        assert [h.label for h in hazards] == [label, 'Object'], label


def test_far_person_does_not_mute_near_person_on_same_side():
    # With these sizes a side person must be near to pass, so use the path:
    # a farther person just left of centre and a near one, same position bucket.
    far, near = det('People', .4, .45, width=.05), det('People', .55, .70, width=.05)
    hazards = run(far, near)
    assert all(h.gated and h.position == 'center' for h in hazards)
    assert {h.band for h in hazards} == {'mid-close', 'near'}
    cooldown = AlertCooldown.from_gate(GATE, clock=lambda: 0.)
    by_band = {h.band: h for h in hazards}
    assert cooldown.should_announce(by_band['mid-close'], now=0.)
    assert cooldown.should_announce(by_band['near'], now=0.5)
    # A truly far person never reaches the cooldown at all.
    assert not run(det('People', .1, .1))[0].gated


def test_local_and_cloud_results_get_identical_gating():
    from detection import DetectionBatch
    detections = (det('Stair', .5, .40, .9), det('People', .8, .3, .7), det('Object', .5, .5, .7))
    local = DetectionBatch(detections, 40., 'yolov8n')
    cloud = DetectionBatch.from_dict(
        {'detections': [d.to_dict() for d in detections if d.class_id >= 0],
         'latency_ms': 140., 'model': 'rfdetr-nano'}, CLASS_NAMES, (W, H))
    cloud_all = cloud.detections + tuple(d for d in detections if d.class_id < 0)
    as_tuples = lambda hs: [(h.label, h.position, h.band, h.gated, h.reason, round(h.priority, 9)) for h in hs]
    assert as_tuples(run(*local.detections)) == as_tuples(run(*cloud_all))


def test_general_person_is_gated_as_people_but_keeps_its_source():
    [hazard] = run(det('People', .5, .65, general=True))
    assert hazard.gated and hazard.source == 'general' and hazard.class_id == -1
    assert phrase_for(hazard) == 'Warning, People ahead'
    [hazard] = run(det('People', .5, .65, confidence=.4, general=True))
    assert hazard.reason == 'low_confidence'


def test_general_detections_use_the_longer_cooldown():
    cooldown = AlertCooldown.from_gate(GATE)
    obj = run(det('Object', .5, .40, confidence=.6))[0]
    stair = run(det('Stair', .5, .40))[0]
    assert cooldown.should_announce(obj, now=0.) and cooldown.should_announce(stair, now=0.)
    assert cooldown.should_announce(stair, now=4.5)
    assert not cooldown.should_announce(obj, now=4.5)
    assert cooldown.should_announce(obj, now=8.5)


def test_approaching_hazard_realerts_before_cooldown_ends():
    cooldown = AlertCooldown.from_gate(GATE)
    first = run(det('Stair', .5, .21))[0]       # mid-close band, proximity 0.60
    closer = run(det('Stair', .5, .33))[0]      # same band, proximity 0.94
    assert first.band == closer.band == 'mid-close'
    assert cooldown.should_announce(first, now=0.)
    assert cooldown.should_announce(closer, now=1.)


def test_disabled_gate_keeps_the_old_behaviour():
    gate = dict(GATE, enabled=False)
    [hazard] = run(det('Stair', .5, .05), gate=gate)
    assert hazard.gated and hazard.band is None
    assert phrase_for(hazard) == 'Note, Stair ahead; distance unavailable'


def test_gate_config_is_validated():
    validate_alert_gate(GATE)
    for broken in ({'min_size': {**GATE['min_size'], 'People': 1.5}},
                   {'near_size': {**GATE['near_size'], 'People': 0.1}},
                   {'min_size': {k: v for k, v in GATE['min_size'].items() if k != 'Object'}},
                   {'object_min_confidence': 0},
                   {'centre_third': [0.7, 0.3]},
                   {'enabled': 'yes'}):
        gate = copy.deepcopy(GATE)
        gate.update(broken)
        with pytest.raises(ValueError):
            validate_alert_gate(gate)
