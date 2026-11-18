"""Evaluate a fixed scale=.75, extra-pillar confidence=.5 experiment on validation."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from config import load_config
from path_utils import read_bgr
from pillar_context import context_input,original_box,merge_pillars


def main():
    import torch
    from ultralytics import YOLO
    from evaluation import load_ground_truth,score_detections
    torch.set_num_threads(4)
    config=load_config()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['context','binary','binary-hard-negative'],default='context')
    args=parser.parse_args()
    if args.method.startswith('binary'):
        from validate_sg_indoor import BinaryColumns
        folder='column-specialist-data' if args.method=='binary' else 'column-specialist-hard-negatives'
        model=BinaryColumns(ROOT/('artifacts/'+folder+'/pilot/weights/best.pt'))
    else:model=YOLO(str(ROOT/'models/local/yolov8n_v4.pt'))
    baseline=json.loads((ROOT/'artifacts/validation-baseline.json').read_text())
    gt,paths=load_ground_truth(ROOT/'data/v4_scene_split/valid',config['classes'])
    out=ROOT/('artifacts/sg-indoor-validation/'+args.method+'-validation');out.mkdir(exist_ok=True)
    outputs={};timings=[];extra_count=0
    for i,path in enumerate(paths):
        frame=read_bgr(path);started=time.perf_counter()
        extras=[]
        if args.method.startswith('binary'):
            extras=[d.to_dict() for d in model.predict(frame).detections
                    if d.box_xyxy[0]>2 and d.box_xyxy[2]<frame.shape[1]-2]
        else:
            canvas,t=context_input(frame)
            result=model.predict(canvas,imgsz=640,conf=.5,iou=.7,classes=[2],rect=False,
                device='cpu',verbose=False)[0]
            for x1,y1,x2,y2,c,cid in result.boxes.data.tolist():
                box=original_box([x1,y1,x2,y2],t)
                if box is not None:extras.append(dict(class_id=2,**{'class':'Pillar'},confidence=c,box_xyxy=box))
        timings.append(1000*(time.perf_counter()-started));extra_count+=len(extras)
        original=baseline[path.name]
        outputs[path.name]=dict(original,detections=merge_pillars(original['detections'],extras),
            model=original.get('model','yolov8n')+' + experimental-context')
        if (i+1)%100==0:print('CONTEXT_VALIDATION',i+1,len(paths),flush=True)
    (out/'predictions.json').write_text(json.dumps(outputs))
    before=score_detections(gt,paths,baseline,config['classes'],.35,config['safety_classes'])
    after=score_detections(gt,paths,outputs,config['classes'],.35,config['safety_classes'])
    report=dict(method=args.method,scale=.75 if args.method=='context' else None,
        horizontal_edge_filter_px=2 if args.method.startswith('binary') else None,
        extra_confidence=.5,validation_images=len(paths),test_used=False,
        application_changed=False,before=before,after=after,total_extra_predictions=extra_count,
        mean_extra_inference_ms=sum(timings)/len(timings),
        note='Exploratory validation, not a fresh independent field test. Context padding reuses the original model; binary methods add a separately trained experimental specialist.')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
