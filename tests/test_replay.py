"""The replay tool's gate re-check catches recorded output the gate would not produce."""
import importlib.util
from pathlib import Path

from config import load_config

spec = importlib.util.spec_from_file_location(
    'replay_alerts', Path(__file__).resolve().parents[1] / 'scripts' / 'replay_alerts.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
CONFIG = load_config()


def record(gated, reason, band):
    return dict(frame=0, width=1280, height=720, detections=[dict(
        class_id=3, **{'class': 'Stair'}, confidence=.9, box_xyxy=[600, 432, 700, 720],
        position='center', band=band, gated=gated, reason=reason)])


def test_gate_recheck_accepts_true_output_and_flags_tampered_output():
    gate, severity = CONFIG['alert_gate'], CONFIG['severity']
    assert replay.regate_mismatches([record(True, None, 'near')  # 288 px of 720 = 0.40 >= near 0.35], gate, severity) == []
    assert replay.regate_mismatches([record(False, 'far', 'far')], gate, severity) == [0]
