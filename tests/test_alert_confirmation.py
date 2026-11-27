"""K-of-N frame confirmation: a hazard is spoken only after it has been seen in K of the last N detector frames."""
import argparse

import pytest

from config import add_detector_arguments, detector_settings, load_config, parse_confirm, validate_alert_gate
from decision_engine import AlertPolicy
from detection import Detection

W, H = 1280, 720
CONFIG = load_config()


def box(x_centre, height, width=.1):
    cx, w, h = x_centre * W, width * W, height * H
    return (cx - w / 2, H - h, cx + w / 2, H)


STAIR = Detection(2, 'Stair', .8, box(.5, .4))
STAIR_MOVED = Detection(2, 'Stair', .8, box(.53, .42))      # same stair, camera moved slightly
STAIR_ELSEWHERE = Detection(2, 'Stair', .8, box(.2, .4))     # a different place in the frame
SIGN = Detection(3, 'Wet Floor Sign', .9, box(.5, .3))


def policy(required=1, window=1):
    gate = dict(CONFIG['alert_gate'], confirm_frames={'required': required, 'window': window})
    return AlertPolicy(gate, CONFIG['severity'], clock=lambda: 0.)


def spoken(p, frames):
    return [p.process(dets, (W, H), now=float(t * 10)).speak for t, dets in enumerate(frames)]


def test_default_config_confirms_two_of_three():
    assert CONFIG['alert_gate']['confirm_frames'] == {'required': 2, 'window': 3}
    p = AlertPolicy(CONFIG['alert_gate'], CONFIG['severity'], clock=lambda: 0.)
    assert spoken(p, [(STAIR,), (STAIR_MOVED,)]) == [None, 'Warning, Stair ahead']


def test_one_of_one_is_the_old_behaviour():
    assert spoken(policy(1, 1), [(STAIR,)])[0] == 'Warning, Stair ahead'


def test_single_frame_flicker_is_never_spoken_with_two_of_three():
    assert spoken(policy(2, 3), [(STAIR,), (), (), (SIGN,), ()]) == [None] * 5


def test_two_consecutive_frames_confirm_and_speak_on_the_second():
    assert spoken(policy(2, 3), [(STAIR,), (STAIR_MOVED,)]) == [None, 'Warning, Stair ahead']


def test_one_missed_frame_in_the_window_still_confirms():
    assert spoken(policy(2, 3), [(STAIR,), (), (STAIR,)]) == [None, None, 'Warning, Stair ahead']


def test_detections_older_than_the_window_do_not_count():
    assert spoken(policy(2, 3), [(STAIR,), (), (), (STAIR,)]) == [None] * 4


def test_a_box_elsewhere_in_the_frame_is_not_the_same_hazard():
    assert spoken(policy(2, 3), [(STAIR_ELSEWHERE,), (STAIR,)]) == [None, None]


def test_banner_still_shows_unconfirmed_hazards():
    result = policy(2, 3).process((STAIR,), (W, H), now=0.)
    assert result.speak is None and result.banner == 'Warning, Stair ahead'


def test_reset_forgets_earlier_frames():
    p = policy(2, 3)
    p.process((STAIR,), (W, H), now=0.)
    p.reset()
    assert p.process((STAIR,), (W, H), now=10.).speak is None


@pytest.mark.parametrize('text, expected', [('2/3', {'required': 2, 'window': 3}), ('1/1', {'required': 1, 'window': 1})])
def test_parse_confirm(text, expected):
    assert parse_confirm(text) == expected


@pytest.mark.parametrize('text', ['3/2', '0/3', 'two/three', '2', '2/11'])
def test_parse_confirm_rejects_bad_values(text):
    with pytest.raises(ValueError):
        parse_confirm(text)


@pytest.mark.parametrize('value', [{'required': 3, 'window': 2}, {'required': 0, 'window': 1},
                                   {'required': True, 'window': 2}, {'required': 2}])
def test_gate_validation_rejects_bad_confirm_frames(value):
    with pytest.raises(ValueError):
        validate_alert_gate(dict(CONFIG['alert_gate'], confirm_frames=value), CONFIG['classes'])


def test_cli_flag_sets_confirm_frames():
    parser = argparse.ArgumentParser()
    add_detector_arguments(parser)
    assert detector_settings(parser.parse_args([]))['alert_gate']['confirm_frames'] == {'required': 2, 'window': 3}
    settings = detector_settings(parser.parse_args(['--confirm', '1/1']))     # --confirm 1/1 turns it off
    assert settings['alert_gate']['confirm_frames'] == {'required': 1, 'window': 1}
    assert spoken(AlertPolicy(settings['alert_gate'], CONFIG['severity'], clock=lambda: 0.), [(STAIR,)]) == ['Warning, Stair ahead']
