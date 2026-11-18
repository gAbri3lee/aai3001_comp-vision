import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_validation_models import matched_truths, paired_interval


def test_duplicate_predictions_cannot_match_the_same_truth_twice():
    truth = [(2,[0,0,10,10]),(2,[20,20,30,30])]
    detections = [dict(class_id=2,confidence=.9,box_xyxy=[0,0,10,10]),
                  dict(class_id=2,confidence=.8,box_xyxy=[0,0,10,10])]
    assert matched_truths(truth,detections,2)=={0}


def test_paired_bootstrap_identical_outputs_have_zero_delta():
    before = [[2,1,1],[1,0,2],[0,1,0]]
    interval = paired_interval(before,before,repeats=100)
    assert interval['recall_delta_95_interval']==[0.,0.]
    assert interval['precision_delta_95_interval']==[0.,0.]
    with pytest.raises(ValueError):
        paired_interval(before,[[1,2,3]],repeats=100)
