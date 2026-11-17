from pathlib import Path
import sys
import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
from pillar_context import context_input,original_box,merge_pillars


def test_context_coordinates_return_to_original_and_reject_padding():
    frame=np.zeros((900,600,3),np.uint8)
    _,t=context_input(frame)
    box=[t['left'],t['top'],t['left']+t['resized_width'],t['top']+t['resized_height']]
    assert original_box(box,t)==[0.,0.,600.,900.]
    assert original_box([0,0,10,10],t) is None


def test_pillar_only_merge_preserves_other_classes_and_deduplicates():
    person=dict(class_id=1,confidence=.8,box_xyxy=[0,0,30,50])
    column=dict(class_id=2,confidence=.6,box_xyxy=[40,0,70,100])
    duplicate=dict(column,confidence=.7)
    result=merge_pillars([person,column],[duplicate,dict(person,confidence=.95)])
    assert person in result and len(result)==2
    assert next(d for d in result if d['class_id']==2)['confidence']==.7
