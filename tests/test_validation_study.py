"""Decision and matching regressions for the validation-only analysis."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('validation_study',
    Path(__file__).resolve().parents[1] / 'scripts/validation_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_duplicate_predictions_and_background_are_false_positives():
    detection = dict(class_id=2, confidence=.8, box_xyxy=[0, 0, 20, 20])
    assert study.count_matches([(2, [0, 0, 20, 20])], [detection, detection], 2, .35) == (1, 1, 0)
    assert study.count_matches([], [detection], 2, .35) == (0, 1, 0)


def test_failed_pilot_precision_is_not_accepted_for_small_recall_gain():
    baseline = dict(precision=47/60, recall=47/161)
    candidate = dict(precision=52/94, recall=52/161, threshold=.35)
    assert study.choose_candidate([candidate], baseline) is None


def test_operating_point_requires_both_precision_and_recall_gain():
    baseline = dict(precision=.80, recall=.30)
    good = dict(precision=.79, recall=.40, threshold=.25)
    bad = dict(precision=.76, recall=.60, threshold=.10)
    assert study.choose_candidate([good, bad], baseline) == good
