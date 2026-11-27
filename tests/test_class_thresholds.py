"""Per-class confidence overrides (--class-conf CLASS=VALUE), opt-in: the shipped thresholds stay unchanged."""
import argparse

import pytest

from config import add_detector_arguments, detector_settings, load_config, parse_class_conf

CONFIG = load_config()


def settings(*argv):
    parser = argparse.ArgumentParser()
    add_detector_arguments(parser)
    return detector_settings(parser.parse_args(list(argv)))


def test_no_flag_keeps_the_shipped_thresholds():
    s = settings()
    assert s['class_thresholds'] == CONFIG['local']['class_thresholds']
    assert s['confidence'] == min(CONFIG['local']['class_thresholds'].values())
    assert s['confidence_override'] is None


def test_lowering_two_classes_leaves_the_others_and_lowers_the_prefilter():
    s = settings('--class-conf', 'Stair=0.25', '--class-conf', 'Wet Floor Sign=0.3')
    expected = dict(CONFIG['local']['class_thresholds'], Stair=.25, **{'Wet Floor Sign': .3})
    assert s['class_thresholds'] == expected
    assert s['confidence'] == .25          # NMS must keep the lowered class's boxes


def test_class_conf_applies_after_global_conf():
    s = settings('--conf', '.5', '--class-conf', 'stair=0.25')
    assert s['class_thresholds'] == {'Escalator': .5, 'People': .5, 'Stair': .25, 'Wet Floor Sign': .5}
    assert s['confidence'] == .25


def test_cloud_requests_the_lowest_threshold_and_filters_per_class_locally():
    s = settings('--backend', 'cloud', '--endpoint', 'http://127.0.0.1:1', '--class-conf', 'Stair=0.2')
    assert s['confidence_override'] == .2
    assert s['class_thresholds']['Stair'] == .2
    assert s['class_thresholds']['People'] == CONFIG['cloud']['class_thresholds']['People']


def test_global_conf_alone_is_unchanged():
    s = settings('--conf', '.6')
    assert s['confidence'] == s['confidence_override'] == .6


def test_class_names_ignore_case_and_spaces():
    assert parse_class_conf([' wet floor sign =0.3']) == {'Wet Floor Sign': .3}


@pytest.mark.parametrize('item', ['Stair', 'Pillar=0.3', 'Stair=abc', 'Stair=1.5', 'Stair=-0.1', 'Stair=nan', '=0.3'])
def test_bad_values_are_rejected(item):
    with pytest.raises(ValueError):
        parse_class_conf([item])
