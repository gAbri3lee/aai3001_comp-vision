"""Real mall primary-target checks; incomplete labels do not support mAP/FP rates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from config import add_detector_arguments, detector_settings
from decision_engine import assess
from detection import Detection, DetectionBatch
from golden import iou
from path_utils import read_bgr

PROMPTS=['column','pillar','person','wall','door','sign','escalator','stairs',
         'chair','bench','trash can','potted plant','suitcase','shopping cart']
MAPPING={0:(2,'Pillar'),1:(2,'Pillar'),2:(1,'People'),6:(0,'Escalator'),7:(3,'Stair'),
         **{i:(-1,'Object') for i in range(8,14)}}


def match_targets(targets,detections,threshold=.5):
    """One-to-one label/IoU assignment for primary targets, not exhaustive GT."""
    candidates=sorted(((iou(t['box_xyxy'],d.box_xyxy),i,j)
        for i,t in enumerate(targets) for j,d in enumerate(detections)
        if t['label']==d.label),reverse=True)
    used_targets,used_predictions=set(),set()
    matches={}
    for overlap,i,j in candidates:
        if overlap>=threshold and i not in used_targets and j not in used_predictions:
            used_targets.add(i);used_predictions.add(j);matches[i]=(j,overlap)
    return matches


class MallSpecialist:
    """Experimental pretrained supplement, not the team's trained checkpoint."""
    def __init__(self,weights,device='cpu'):
        from ultralytics import YOLOE
        self.model=YOLOE(str(weights));self.device=device
        if list(self.model.names.values())!=PROMPTS:
            raise ValueError('Frozen specialist vocabulary mismatch')

    def predict(self,frame):
        started=time.perf_counter()
        r=self.model.predict(frame,imgsz=640,conf=.35,iou=.7,rect=False,
            device=self.device,verbose=False)[0]
        detections=[]
        for x1,y1,x2,y2,confidence,cid in sorted(r.boxes.data.cpu().tolist(),key=lambda x:-x[4]):
            if int(cid) not in MAPPING:continue
            class_id,label=MAPPING[int(cid)]
            box=(max(0,x1),max(0,y1),min(frame.shape[1],x2),min(frame.shape[0],y2))
            if box[0]>=box[2] or box[1]>=box[3]:continue
            d=Detection(class_id,label,confidence,box)
            if any(old.label==d.label and iou(old.box_xyxy,d.box_xyxy)>=.5 for old in detections):continue
            detections.append(d)
        return DetectionBatch(tuple(detections),1000*(time.perf_counter()-started),'experimental-yoloe-26s-mall')


class VisualPillar:
    def __init__(self,weights):
        from ultralytics import YOLOE
        self.model=YOLOE(str(weights))

    def predict(self,frame):
        started=time.perf_counter()
        r=self.model.predict(frame,imgsz=640,conf=.35,iou=.7,rect=False,device='cpu',verbose=False)[0]
        ds=tuple(Detection(2,'Pillar',float(row[4]),tuple(map(float,row[:4])))
                 for row in r.boxes.data.cpu().tolist())
        return DetectionBatch(ds,1000*(time.perf_counter()-started),'experimental-visual-pillar')


class GroundedMall:
    TEXT='a building column. a pillar. a trash can. a chair. a potted plant. a person. an escalator. stairs.'

    def __init__(self,folder):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.processor=AutoProcessor.from_pretrained(str(folder),local_files_only=True)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(str(folder),local_files_only=True).eval()

    def predict(self,frame):
        import torch
        from PIL import Image
        started=time.perf_counter()
        inputs=self.processor(images=Image.fromarray(frame[:,:,::-1]),text=self.TEXT,return_tensors='pt')
        with torch.inference_mode():outputs=self.model(**inputs)
        r=self.processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
            threshold=.35,text_threshold=.25,target_sizes=[frame.shape[:2]])[0]
        ds=[]
        candidates=zip(r['boxes'],r['scores'],r['text_labels'])
        for box,score,label in sorted(candidates,key=lambda item:-float(item[1])):
            if 'column' in label or 'pillar' in label:cid,name=2,'Pillar'
            elif 'person' in label:cid,name=1,'People'
            elif 'escalator' in label:cid,name=0,'Escalator'
            elif 'stairs' in label:cid,name=3,'Stair'
            elif any(s in label for s in ['trash','chair','plant']):cid,name=-1,'Object'
            else:continue
            b=box.tolist();b=(max(0.,b[0]),max(0.,b[1]),min(float(frame.shape[1]),b[2]),min(float(frame.shape[0]),b[3]))
            if b[0]>=b[2] or b[1]>=b[3]:continue
            d=Detection(cid,name,float(score),b)
            if any(old.label==name and iou(old.box_xyxy,b)>=.5 for old in ds):continue
            ds.append(d)
        return DetectionBatch(tuple(ds),1000*(time.perf_counter()-started),'experimental-grounding-dino-tiny')


class BinaryColumns:
    def __init__(self,weights):
        from ultralytics import YOLO
        self.model=YOLO(str(weights))
        if list(self.model.names.values())!=['structural column']:raise ValueError('Wrong binary checkpoint')

    def predict(self,frame):
        started=time.perf_counter()
        result=self.model.predict(frame,imgsz=416,conf=.5,iou=.7,rect=False,device='cpu',verbose=False)[0]
        ds=tuple(Detection(2,'Pillar',float(row[4]),tuple(map(float,row[:4])))
                 for row in result.boxes.data.cpu().tolist())
        return DetectionBatch(ds,1000*(time.perf_counter()-started),'experimental-trained-column-specialist')


def main():
    import cv2
    import numpy as np
    import torch
    from scene_detector import SceneDetector
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['baseline','specialist','visual','grounding','binary','binary-hard-negative'],required=True)
    parser.add_argument('--role',choices=['development','heldout','all'],default='all')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=ROOT/'annotations/sg_indoor_primary_targets.json')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    config=argparse.ArgumentParser();add_detector_arguments(config)
    settings=detector_settings(config.parse_args([]))
    torch.set_num_threads(4)
    paths=dict(baseline='models/local/yolov8n_v4.pt',specialist='artifacts/sg-pillar-specialist/yoloe-26s-mall-fixed.pt',
        visual='artifacts/sg-pillar-specialist/yoloe-26s-visual-pillar.pt',grounding='artifacts/sg-pillar-specialist/grounding-dino-tiny',
        binary='artifacts/column-specialist-data/pilot/weights/best.pt',
        **{'binary-hard-negative':'artifacts/column-specialist-hard-negatives/pilot/weights/best.pt'})
    weights=ROOT/paths[args.arm]
    if args.arm=='baseline':model=SceneDetector(settings)
    elif args.arm=='visual':model=VisualPillar(weights)
    elif args.arm=='grounding':model=GroundedMall(weights)
    elif args.arm.startswith('binary'):model=BinaryColumns(weights)
    else:model=MallSpecialist(weights)
    model.predict(np.zeros((480,640,3),dtype=np.uint8))
    payload=json.loads(args.manifest.read_text())
    rows=[r for r in payload['images'] if args.role=='all' or
          (r['role'].startswith('development'))==(args.role=='development')]
    weight_file=weights/'model.safetensors' if weights.is_dir() else weights
    report=dict(arm=args.arm,role=args.role,model_sha256=hashlib.sha256(weight_file.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        source_scope=payload['scope'],no_physical_distance_ground_truth=True,
        specialist_is_pretrained_not_team_model=args.arm in ('specialist','visual','grounding'),
        specialist_is_small_experimental_training=args.arm.startswith('binary'),images=[])
    for row in rows:
        path=ROOT/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source changed')
        frame=read_bgr(path);started=time.perf_counter();batch=model.predict(frame)
        elapsed=1000*(time.perf_counter()-started)
        hazards=assess(batch.detections,(frame.shape[1],frame.shape[0]),settings['alert_gate'],settings['severity'])
        by_index={h.index:h for h in hazards}
        matched=match_targets(row['targets'],batch.detections)
        results=[]
        for i,target in enumerate(row['targets']):
            j,overlap=matched.get(i,(None,None))
            results.append(dict(target,matched=j is not None,matched_prediction=j,iou=overlap,
                alert_eligible=by_index[j].gated if j is not None else False,
                alert_reason=by_index[j].reason if j is not None else 'missed'))
        record=dict(id=row['id'],venue=row['venue'],role=row['role'],elapsed_ms=elapsed,
            detections=[d.to_dict() for d in batch.detections],hazards=[h.to_dict() for h in hazards],
            targets=results,general_error=getattr(model,'general_error',None))
        report['images'].append(record)
        for d in batch.detections:
            x1,y1,x2,y2=map(int,d.box_xyxy);cv2.rectangle(frame,(x1,y1),(x2,y2),(0,160,255),2)
            cv2.putText(frame,f'{d.label} {d.confidence:.2f}',(x1,max(18,y1)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,160,255),2)
        for t in results:
            x1,y1,x2,y2=map(int,t['box_xyxy']);cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0) if t['matched'] else (0,0,255),3)
        cv2.imwrite(str(args.output/(row['id']+'.jpg')),frame)
        print('MALL_IMAGE',row['id'],[(t['label'],t['matched'],t['alert_eligible']) for t in results],flush=True)
    summary={}
    for label in ['Pillar','Object','People','Escalator','Stair','Wet Floor Sign']:
        targets=[t for r in report['images'] for t in r['targets'] if t['label']==label]
        summary[label]=dict(targets=len(targets),matched=sum(t['matched'] for t in targets),
            alert_eligible=sum(t['alert_eligible'] for t in targets))
    times=[r['elapsed_ms'] for r in report['images']]
    report.update(primary_target_summary=summary,median_inference_ms=statistics.median(times),
        p95_inference_ms=float(np.percentile(times,95)),blank_controls=[])
    for name,frame in [('black',np.zeros((480,640,3),np.uint8)),('white',np.full((480,640,3),255,np.uint8))]:
        batch=model.predict(frame);report['blank_controls'].append(dict(name=name,detections=[d.to_dict() for d in batch.detections]))
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(summary=summary,median_inference_ms=report['median_inference_ms'])),flush=True)


if __name__=='__main__':main()
