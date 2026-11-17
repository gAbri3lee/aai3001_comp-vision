from pathlib import Path
import sys

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
from detection import Detection
from validate_sg_indoor import match_targets


def test_a_prediction_cannot_count_as_two_primary_targets():
    t=[dict(label='Pillar',box_xyxy=[0,0,20,100])]*2
    assert len(match_targets(t,[Detection(2,'Pillar',.8,(0,0,20,100))]))==1


def test_wrong_classes_and_weak_overlap_do_not_count_as_recognition():
    targets=[dict(label='Pillar',box_xyxy=[0,0,20,100])]
    assert not match_targets(targets,[Detection(-1,'Object',.9,(0,0,20,100))])
    assert not match_targets(targets,[Detection(2,'Pillar',.9,(19,0,39,100))])
