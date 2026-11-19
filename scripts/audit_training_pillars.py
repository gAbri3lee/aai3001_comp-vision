"""TRAIN-only annotation proposals from two detectors; never edits labels.

High-confidence boxes unmatched to existing Pillar labels are review candidates,
not ground truth. Uses the prepared private v4 input on Kaggle, excludes test.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import time

PROMPTS = ['pillar','column','bollard','pole','person','tree','wall','cabinet','sign','door','escalator','stairs','chair']


def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union>0 else 0


def main():
    import torch
    from ultralytics import YOLO, YOLOWorld
    assert torch.cuda.is_available(),'GPU required'
    roots=list(Path('/kaggle/input').rglob('pilot_input.json'))
    assert len(roots)==1
    root=roots[0].parent
    assert not (root/'test').exists()
    manifest=(root/'split_manifest.csv').read_bytes()
    assert hashlib.md5(manifest).hexdigest()=='49601080aac1f838190341d331785578'
    scenes={r['stem']:r['scene'] for r in csv.DictReader(manifest.decode().splitlines()) if r['split']=='train'}
    paths=sorted((root/'train/images').glob('*'))
    assert len(paths)==8247 and {p.stem for p in paths}==set(scenes)
    output=Path('/kaggle/working/training_annotation_audit')
    output.mkdir(exist_ok=True)
    cache=Path('/tmp/visionaid_annotation_models')
    cache.mkdir(exist_ok=True)
    os.chdir(cache)
    candidates=[]
    def run(model,source):
        count=0
        for prediction in model.predict(source=str(root/'train/images'),stream=True,batch=16,
            imgsz=640,conf=.35 if source=='world' else .5,iou=.7,rect=False,
            agnostic_nms=source=='world',device=0,verbose=False):
            path=Path(prediction.path)
            assert path.stem in scenes
            h,w=prediction.orig_shape
            label=root/'train/labels'/(path.stem+'.txt')
            truths=[]
            for line in label.read_text().splitlines() if label.exists() else []:
                cid,x,y,bw,bh=map(float,line.split())
                truths.append((int(cid),[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
            for row in prediction.boxes.data.cpu().tolist():
                cid=int(row[5])
                if (source=='world' and cid not in (0,1,2,3)) or (source=='trained' and cid!=2):
                    continue
                box=row[:4]
                overlap=max((iou(box,b) for c,b in truths if c==2),default=0)
                if overlap>=.3:
                    continue
                candidates.append(dict(image=path.name,scene=scenes[path.stem],source=source,
                    prompt=PROMPTS[cid] if source=='world' else 'Pillar',confidence=row[4],
                    box_xyxy=box,width=w,height=h,pillar_label_iou=overlap,
                    other_label_iou=max((iou(box,b) for c,b in truths if c!=2),default=0),
                    status='unreviewed_detector_proposal_not_ground_truth'))
            count+=1
            if count%500==0:
                print('AUDIT_PROGRESS',source,count,'proposals',len(candidates),flush=True)
        assert count==8247
        (output/f'{source}_complete.json').write_text(json.dumps(dict(images=count,proposals=candidates),indent=2))
    trained=YOLO(str(root/'yolov8n_v4.pt'))
    metadata=json.loads(roots[0].read_text())
    assert hashlib.sha256((root/'yolov8n_v4.pt').read_bytes()).hexdigest()==metadata['weights_sha256']
    print('AUDIT_GPU',torch.cuda.get_device_name(0),flush=True)
    run(trained,'trained')
    del trained
    torch.cuda.empty_cache()
    world=YOLOWorld(str(cache/'yolov8s-worldv2.pt'))
    world.set_classes(PROMPTS)
    run(world,'world')
    report=dict(training_images=8247,prompts=PROMPTS,proposals=candidates,test_used=False,
        validation_used=False,labels_changed=False,
        note='Model proposals require full-image visual review. Never auto-add these as annotations.')
    (output/'proposals.json').write_text(json.dumps(report,indent=2))
    print('AUDIT_COMPLETE',json.dumps(dict(images=8247,proposals=len(candidates))),flush=True)


if __name__=='__main__':
    main()
