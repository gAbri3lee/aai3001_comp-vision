"""Independently compare native predictions on the unchanged v4 validation split.

Reports paired scene-bootstrap uncertainty and size-binned truth recall. Never
reads test images, tunes thresholds, edits labels, or promotes weights.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from config import CLASS_NAMES, load_config
from detection import Detection
from decision_engine import assess
from evaluation import load_ground_truth, score_detections
from golden import iou
from validation_study import count_matches


def matched_truths(truth, detections, cid, threshold=.35):
    boxes = [box for label,box in truth if label==cid]
    matched = set()
    for detection in sorted((d for d in detections if d['class_id']==cid and d['confidence']>=threshold),
                            key=lambda d:-d['confidence']):
        available = [k for k in range(len(boxes)) if k not in matched]
        best = max(available,key=lambda k:iou(detection['box_xyxy'],boxes[k]),default=None)
        if best is not None and iou(detection['box_xyxy'],boxes[best])>=.5:
            matched.add(best)
    return matched


def paired_interval(before,after,seed=0,repeats=5000):
    import numpy as np
    before,after = np.asarray(before),np.asarray(after)
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 3:
        raise ValueError('Need aligned scene TP/FP/FN arrays')
    draws = np.random.default_rng(seed).integers(0,len(before),(repeats,len(before)))
    b,a = before[draws].sum(1),after[draws].sum(1)
    report = {}
    for metric,column in [('precision',1),('recall',2)]:
        supported = ((b[:,0]+b[:,column])>0)&((a[:,0]+a[:,column])>0)
        delta = a[supported,0]/(a[supported,0]+a[supported,column])-b[supported,0]/(b[supported,0]+b[supported,column])
        report[metric+'_delta_95_interval'] = np.quantile(delta,[.025,.975]).tolist() if len(delta) else None
    return report


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/v4_scene_split')
    parser.add_argument('--baseline',type=Path,default=ROOT/'artifacts/validation-baseline.json')
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = (args.dataset/'split_manifest.csv').read_bytes()
    if hashlib.md5(manifest).hexdigest() != '49601080aac1f838190341d331785578':
        raise ValueError('Unexpected validation split')
    rows = list(csv.DictReader(manifest.decode().splitlines()))
    scenes = {r['stem']:r['scene'] for r in rows if r['split'] in ('val','valid')}
    names = sorted(set(scenes.values()))
    indices = {name:i for i,name in enumerate(names)}
    gt,paths = load_ground_truth(args.dataset/'valid',CLASS_NAMES)
    if set(scenes)!={p.stem for p in paths}:
        raise ValueError('Validation manifest mismatch')
    baseline,candidate = json.loads(args.baseline.read_text()),json.loads(args.candidate.read_text())
    reports = [score_detections(gt,paths,outputs,CLASS_NAMES,.35,['Escalator','Stair','Wet Floor Sign'])
               for outputs in (baseline,candidate)]
    truths = defaultdict(list)
    for annotation in gt['annotations']:
        x,y,w,h = annotation['bbox']
        truths[annotation['image_id']].append((annotation['category_id']-1,[x,y,x+w,y+h]))
    grouped = np.zeros((2,len(names),5,3),dtype=np.int64)
    config = load_config()
    gate_totals = dict(eligible_truths=0,baseline_detected_and_gated=0,candidate_detected_and_gated=0)
    sizes = {name:dict(boxes=0,baseline_tp=0,candidate_tp=0) for name in ('small','medium','large')}
    for index,path in enumerate(paths):
        truth = truths[index]
        shape = (gt['images'][index]['width'],gt['images'][index]['height'])
        pillar_truths = [(cid,box) for cid,box in truth if cid==2]
        truth_hazards = assess([Detection(cid,CLASS_NAMES[cid],1.,tuple(box)) for cid,box in pillar_truths],
                              shape,config['alert_gate'],config['severity'])
        eligible = {h.index for h in truth_hazards if h.gated}
        gate_totals['eligible_truths'] += len(eligible)
        for model_index,outputs in enumerate((baseline,candidate)):
            detections = outputs[path.name]['detections']
            for cid in range(5):
                grouped[model_index,indices[scenes[path.stem]],cid] += count_matches(truth,detections,cid,.35)
            matched = matched_truths(truth,detections,2)
            strong = [d for d in detections if d['confidence']>=.35]
            hazards = assess([Detection(d['class_id'],CLASS_NAMES[d['class_id']],d['confidence'],tuple(d['box_xyxy']))
                              for d in strong],shape,config['alert_gate'],config['severity'])
            gated = [strong[h.index] for h in hazards if h.gated]
            gated_matches = matched_truths(truth,gated,2)
            gate_totals['candidate_detected_and_gated' if model_index else 'baseline_detected_and_gated'] += len(gated_matches&eligible)
            boxes = [box for cid,box in truth if cid==2]
            for k,box in enumerate(boxes):
                size = (box[3]-box[1])/gt['images'][index]['height']
                bucket = sizes['small' if size<.15 else 'medium' if size<.4 else 'large']
                if not model_index:
                    bucket['boxes'] += 1
                bucket['candidate_tp' if model_index else 'baseline_tp'] += k in matched
    before,after = reports
    b,a = before['per_class']['Pillar'],after['per_class']['Pillar']
    regressions = [n for n in CLASS_NAMES if n!='Pillar' and
                   after['per_class'][n]['recall'] < before['per_class'][n]['recall']-.03]
    passed = (a['recall'] >= b['recall']+.10 and a['precision'] >= max(.75,b['precision']-.02)
              and after['map50'] >= before['map50']-.01 and not regressions)
    report = dict(images=len(paths),recorded_scene_groups=len(names),baseline=before,candidate=after,
        gate_passed=passed,other_class_regressions=regressions,pillar_size_bins=sizes,
        pillar_alert_gate=gate_totals,
        paired_scene_bootstrap={n:paired_interval(grouped[0,:,i],grouped[1,:,i]) for i,n in enumerate(CLASS_NAMES)},
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        test_used=False,application_changed=False,threshold=.35,
        limitations=['Validation-selected model; intervals do not account for model-selection optimism',
                     'Recorded groups are not verified independent sites; no domain-shift or physical-distance claim',
                     'Alert-gate counts concern eligibility, not proof of selection, speech playback or safe navigation',
                     'Size bins use box-height fraction: small <0.15, medium <0.4, large >=0.4'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
