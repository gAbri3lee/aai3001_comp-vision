"""Debug-panel distance estimate: pinhole maths, truncation and calibration file."""
import json

import pytest

import distance


def test_focal_and_estimate_round_trip():
    focal = distance.focal_from_measurement(box_height_px=432, distance_m=3.0, real_height_m=1.7)
    calibration = dict(focal_px=focal, frame_height=720)
    value, note = distance.estimate('People', (100, 100, 200, 532), 720, calibration, {'People': 1.7})
    assert note == 'estimate' and value == pytest.approx(3.0)
    # Same camera at half resolution: the focal length in pixels halves too.
    value, _ = distance.estimate('People', (50, 50, 100, 266), 360, calibration, {'People': 1.7})
    assert value == pytest.approx(3.0)


def test_no_estimate_for_cut_off_boxes_unknown_classes_or_uncalibrated():
    calibration = dict(focal_px=900, frame_height=720)
    assert distance.estimate('People', (0, 0, 10, 300), 720, calibration, {'People': 1.7}) == (None, 'box cut off by frame edge')
    assert distance.estimate('People', (0, 300, 10, 720), 720, calibration, {'People': 1.7})[0] is None
    assert distance.estimate('Stair', (0, 100, 10, 300), 720, calibration, {'People': 1.7})[1] == 'no known height for this class'
    assert distance.estimate('People', (0, 100, 10, 300), 720, None, {'People': 1.7}) == (None, 'not calibrated')
    with pytest.raises(ValueError):
        distance.focal_from_measurement(0, 3, 1.7)


def test_calibration_file_uses_the_median_sample(tmp_path):
    path = tmp_path / 'camera_calibration.json'
    assert distance.load_calibration(path) is None
    path.write_text(json.dumps({'samples': [dict(focal_px=f, frame_height=720) for f in (880, 900, 1500)]}))
    assert distance.load_calibration(path) == dict(focal_px=900, frame_height=720, samples=3)


def test_calibration_normalizes_mixed_resolutions_and_rejects_corrupt_files(tmp_path):
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps({'samples': [dict(focal_px=900, frame_height=720),
                                          dict(focal_px=450, frame_height=360)]}))
    assert distance.load_calibration(path)['focal_px'] == pytest.approx(450)
    for data in ('broken json', '[]', '{"samples":[{"focal_px": 2,"frame_height": 0}]}'):
        path.write_text(data)
        assert distance.load_calibration(path) is None
    for value in (float('nan'), float('inf'), -1):
        with pytest.raises(ValueError):
            distance.focal_from_measurement(value, 3, 1.7)


def test_zero_height_box_has_no_distance_estimate():
    assert distance.estimate('People', (10, 100, 20, 100), 720,
                             dict(focal_px=900, frame_height=720), {'People': 1.7}) == (None, 'invalid box')
