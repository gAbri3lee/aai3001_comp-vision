"""Migration boundaries: old cloud IDs must never silently become new hazards."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from requests import Response

from config import CLASS_NAMES, LEGACY_CLASS_NAMES, add_detector_arguments, detector_settings, load_config
from detectors import CloudDetector, DetectorError
from rfdetr_detector import RFDetrDetector
from media import valid_expected


def cloud_settings():
    p=argparse.ArgumentParser();add_detector_arguments(p)
    return detector_settings(p.parse_args(['--backend','cloud','--endpoint','http://localhost:9000']))


def reply(detections):
    r=Response();r.status_code=200
    r._content=json.dumps(dict(model='rfdetr-nano',latency_ms=2,detections=detections)).encode()
    return r


def item(cid,label):
    return dict(class_id=cid,**{'class':label},confidence=.8,box_xyxy=[0,0,20,20])


def test_active_contract_is_dense_four_and_rejects_removed_expected_class():
    assert load_config()['classes']==list(CLASS_NAMES)==['Escalator','People','Stair','Wet Floor Sign']
    assert 'Pillar' not in load_config()['severity']
    with pytest.raises(ValueError):valid_expected('Pillar')


@pytest.mark.parametrize('items,expected',[
    ([item(2,'Pillar'),item(3,'Stair'),item(4,'Wet Floor Sign')],[(2,'Stair'),(3,'Wet Floor Sign')]),
    ([item(2,'Stair'),item(3,'Wet Floor Sign')],[(2,'Stair'),(3,'Wet Floor Sign')]),
    ([item(2,'Pillar')],[]),
])
def test_cloud_native_and_legacy_payloads_map_by_validated_name(items,expected):
    model=CloudDetector(cloud_settings())
    with patch.object(model.session,'post',return_value=reply(items)):
        result=model.predict(np.zeros((40,40,3),dtype=np.uint8))
    assert [(d.class_id,d.label) for d in result.detections]==expected


def test_cloud_rejects_mixed_or_forged_class_order():
    model=CloudDetector(cloud_settings())
    with patch.object(model.session,'post',return_value=reply([item(2,'Stair'),item(4,'Wet Floor Sign')])):
        with pytest.raises(DetectorError):model.predict(np.zeros((40,40,3),dtype=np.uint8))


@pytest.mark.parametrize('names,ok',[(list(CLASS_NAMES),True),(list(LEGACY_CLASS_NAMES),True),
    (['People','Escalator','Stair','Wet Floor Sign'],False)])
def test_cloud_health_only_accepts_declared_contracts(names,ok):
    model=CloudDetector(cloud_settings());response=reply([])
    response._content=json.dumps(dict(status='ready',model='rfdetr-nano',classes=names)).encode()
    with patch('requests.get',return_value=response):
        if ok:assert model.health()['classes']==names
        else:
            with pytest.raises(DetectorError):model.health()


def test_server_legacy_background_and_removed_class_are_filtered_before_remap():
    config=load_config();detector=RFDetrDetector.__new__(RFDetrDetector)
    detector.config,detector.spec=config,config['cloud']
    detector.model=SimpleNamespace(predict=lambda *a,**k:SimpleNamespace(
        class_id=np.array([2,3,4,5]),confidence=np.array([.9,.8,.7,.6]),
        xyxy=np.array([[0,0,20,20]]*4)))
    result=detector.predict(Image.new('RGB',(40,40)))
    assert [(d.class_id,d.label) for d in result.detections]==[(2,'Stair'),(3,'Wet Floor Sign')]


def test_dataset_quarantine_preserves_splits_and_never_keeps_candidate_pair():
    path=Path(__file__).resolve().parents[1]/'scripts/build_four_class_dataset.py'
    spec=importlib.util.spec_from_file_location('four_dataset_build',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[dict(split=s,file=f'{i}.jpg',labels=[[1,.5,.5,.2,.2]],width=640,height=640,pixel_sha256=str(i))
          for i,s in enumerate(['train','valid','test'])]
    audit=dict(rows=rows,byte_duplicate_groups=[],pixel_duplicate_groups=[],perceptual_pairs=[dict(a=0,b=1)])
    kept,excluded=module.select_rows(audit)
    assert kept=={0,2} and excluded[0]['index']==1
    assert [x['split'] for x in rows]==['train','valid','test']
