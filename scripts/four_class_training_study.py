"""Preregistered GPU-only paired four-class fine-tuning; no TEST pixels or tuning.

Three arms separate label repair from scene balancing. All start with the same
native head, use the same epochs/draw count/hyperparameters, and report final
epoch results. No checkpoint is automatically deployed.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

NAMES=['Escalator','People','Stair','Wet Floor Sign']
ARMS=['control','reviewed','reviewed_scene_balanced']


def balanced_draws(rows,seed,total):
    rng=random.Random(seed);negative=[r['stem'] for r in rows if not r['labels']]
    positive=[r for r in rows if r['labels']];groups=[defaultdict(list) for _ in NAMES]
    for r in positive:
        for cid in {int(x[0]) for x in r['labels']}:groups[cid][r['scene']].append(r['stem'])
    counts=Counter();chosen=[]
    for index in range(total-len(negative)):
        cid=index%4;available={s:[x for x in xs if counts[x]<3] for s,xs in groups[cid].items()}
        available={s:xs for s,xs in available.items() if xs}
        if not available:raise ValueError('Image repeat cap exhausted; do not silently change sampling policy')
        scene=rng.choice(sorted(available));stem=rng.choice(available[scene]);chosen.append(stem);counts[stem]+=1
    chosen+=negative;rng.shuffle(chosen)
    assert len(chosen)==total and max(Counter(chosen).values())<=3
    return chosen


def link_image(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        try:destination.symlink_to(source)
        except (OSError,NotImplementedError):shutil.copy2(source,destination)


def main():
    import numpy as np
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--cpu',action='store_true',help='Explicit separate local pilot, never an implicit GPU fallback')
    parser.add_argument('--input',type=Path);parser.add_argument('--image-source',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--resume-study',action='store_true',help='Recover completed final-epoch outputs after an orchestration error')
    args=parser.parse_args()
    if not args.cpu and not torch.cuda.is_available():raise RuntimeError('GPU required; no silent CPU fallback')
    if args.cpu:
        if not all([args.input,args.image_source,args.output]):raise ValueError('Explicit CPU pilot needs separate input/source/output')
        configs=[args.input/'study_plan.json'];source=args.image_source.resolve();torch.set_num_threads(4)
    else:
        configs=list(Path('/kaggle/input').rglob('study_plan.json'));old=list(Path('/kaggle/input').rglob('pilot_input.json'))
        if len(configs)!=1 or len(old)!=1:raise ValueError('Attach precisely the new study input and original private v4 image input')
        source=old[0].parent
    root=configs[0].parent.resolve();plan=json.loads(configs[0].read_text());device='cpu' if args.cpu else 0
    if args.seed not in plan['seeds']:raise ValueError('Unregistered seed')
    if (source/'test').exists():raise ValueError('Private study source must exclude test images')
    records=json.loads((root/'records.json').read_text());repairs=json.loads((root/'review-decisions.json').read_text())['repairs']
    if hashlib.sha256((root/'records.json').read_bytes()).hexdigest()!=plan['records_sha256']:raise ValueError('Records changed')
    if hashlib.sha256((root/'review-decisions.json').read_bytes()).hexdigest()!=plan['review_sha256']:raise ValueError('Review decisions changed')
    weights=root/'yolov8n_v4_four_class.pt'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=plan['weights_sha256']:raise ValueError('Checkpoint changed')
    assert Counter(r['split'] for r in records)==plan.get('split_images',{'train':7749,'valid':859})
    scene_splits=defaultdict(set)
    for r in records:
        scene_splits[r['scene']].add(r['split']);p=source/r['split']/'images'/r['original_file']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Image mismatch '+r['stem'])
    assert all(len(x)==1 for x in scene_splits.values())
    sys.path.insert(0,str(root))
    from evaluation import load_ground_truth,score_detections
    work=args.output.resolve() if args.cpu else Path('/kaggle/working');work.mkdir(parents=True,exist_ok=True)
    out=work/f'four-class-seed{args.seed}'
    if out.exists() and args.resume_study:
        assert json.loads((out/'registered-plan.json').read_text())==plan,'Cannot resume a changed plan'
    out.mkdir(exist_ok=args.resume_study)
    (out/'registered-plan.json').write_text(json.dumps(plan,indent=2));(out/'review-decisions.json').write_text((root/'review-decisions.json').read_text())
    validation=out/'validation'
    for r in records:
        if r['split']!='valid':continue
        link_image(source/'valid/images'/r['original_file'],validation/'images'/(r['stem']+'.jpg'))
        p=validation/'labels'/(r['stem']+'.txt');p.parent.mkdir(exist_ok=True)
        p.write_text(''.join(' '.join(map(str,line))+'\n' for line in r['labels']))
    gt,paths=load_ground_truth(validation,NAMES)
    def evaluate(path,tag):
        model=YOLO(str(path));outputs={}
        for pred in model.predict(source=str(validation/'images'),batch=plan.get('batch',32),stream=True,imgsz=640,conf=.001,iou=.7,max_det=300,rect=False,device=device,verbose=False):
            outputs[Path(pred.path).name]=dict(detections=[dict(class_id=int(c),**{'class':NAMES[int(c)]},confidence=float(s),box_xyxy=[a,b,x,y]) for a,b,x,y,s,c in pred.boxes.data.cpu().tolist()])
        metrics=score_detections(gt,paths,outputs,NAMES,.35,['Stair','Escalator','Wet Floor Sign'])
        (out/(tag+'-predictions.json')).write_text(json.dumps(outputs));(out/(tag+'-metrics.json')).write_text(json.dumps(metrics,indent=2))
        del model;torch.cuda.empty_cache();return metrics
    class PreserveNativeHead(DetectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            if weights is None:raise ValueError('No starting checkpoint supplied')
            model=copy.deepcopy(weights).float();model.criterion=None
            assert model.model[-1].nc==4 and all(x[-1].out_channels==4 for x in model.model[-1].cv3)
            assert all(torch.equal(v,weights.state_dict()[k].float() if v.is_floating_point() else weights.state_dict()[k]) for k,v in model.state_dict().items())
            return model
    if args.cpu and plan.get('reuse_baseline_metrics_sha256'):
        cached=root/'baseline-metrics.json';assert hashlib.sha256(cached.read_bytes()).hexdigest()==plan['reuse_baseline_metrics_sha256']
        baseline=json.loads(cached.read_text())
    else:baseline=evaluate(weights,'baseline')
    summary=dict(seed=args.seed,baseline=baseline,arms={},test_used=False,device='cpu' if args.cpu else torch.cuda.get_device_name(0),plan_sha256=hashlib.sha256(configs[0].read_bytes()).hexdigest())
    if args.resume_study and (out/'summary.json').exists():
        saved=json.loads((out/'summary.json').read_text());assert saved['plan_sha256']==summary['plan_sha256'];summary=saved
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    train=[r for r in records if r['split']=='train'];by_stem={r['stem']:r for r in train}
    for arm in ARMS:
        if arm in summary['arms']:continue
        armroot=out/arm;armroot.mkdir(exist_ok=args.resume_study);start=time.monotonic();armrows=[]
        for r in train:
            labels=repairs.get(r['stem'],{}).get('replacement_labels',r['labels']) if arm!='control' else r['labels']
            armrows.append(dict(r,labels=labels));link_image(source/'train/images'/r['original_file'],armroot/'train/images'/(r['stem']+'.jpg'))
            path=armroot/'train/labels'/(r['stem']+'.txt');path.parent.mkdir(exist_ok=True)
            path.write_text(''.join(' '.join(map(str,line))+'\n' for line in labels))
        draws=balanced_draws(armrows,args.seed,len(train)) if arm=='reviewed_scene_balanced' else [r['stem'] for r in train]
        (armroot/'draws.txt').write_text('\n'.join(draws)+'\n')
        trainlist=armroot/'train.txt';trainlist.write_text(''.join(str(armroot/'train/images'/(s+'.jpg'))+'\n' for s in draws))
        yamlpath=armroot/'data.yaml';yamlpath.write_text(yaml.safe_dump(dict(path=str(armroot),train=str(trainlist),val=str(validation/'images'),nc=4,names=NAMES)))
        completed=list(armroot.glob('*/weights/last.pt')) if args.resume_study else []
        recovered=False
        if completed:
            import csv
            if len(completed)!=1:raise ValueError('Ambiguous recovery checkpoint')
            result_rows=list(csv.DictReader((completed[0].parent.parent/'results.csv').open()))
            if len(result_rows)!=plan['epochs']:raise ValueError('Recovery checkpoint is not the preregistered final epoch')
            last=completed[0];recovered=True
        else:
            model=YOLO(str(weights))
            model.train(trainer=PreserveNativeHead,data=str(yamlpath),epochs=plan['epochs'],patience=1000,imgsz=plan.get('training_input_size',640),batch=plan.get('batch',32),
            device=device,workers=0 if args.cpu else 2,seed=args.seed,deterministic=True,optimizer='AdamW',lr0=.0003,lrf=.2,weight_decay=.0005,
            warmup_epochs=.5,cos_lr=True,mosaic=.3,mixup=0,close_mosaic=1,scale=.25,translate=.1,fliplr=.5,
            hsv_h=.015,hsv_s=.4,hsv_v=.2,project=str(armroot),name='fit',exist_ok=False,save=True,plots=False,
            amp=not args.cpu,cache=False)
            last=Path(model.trainer.last);assert last.is_file()
            del model
        torch.cuda.empty_cache();metrics=evaluate(last,arm)
        summary['arms'][arm]=dict(metrics=metrics,seconds=time.monotonic()-start,draws=len(draws),unique_draws=len(set(draws)),
            maximum_image_repeats=max(Counter(draws).values()),labels_changed=arm!='control',final_checkpoint=str(last.relative_to(out)),recovered_completed_training=recovered)
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print('ARM COMPLETE',args.seed,arm,json.dumps(summary['arms'][arm]),flush=True)
    print('STUDY COMPLETE',args.seed,flush=True)


if __name__=='__main__':main()
