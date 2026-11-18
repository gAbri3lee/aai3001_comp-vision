"""Small exploratory binary-column pilot; uses v4 TRAIN crops only.

No new Singapore challenge photo is used for training or checkpoint selection.
AI-reviewed crops and tiny internal validation are not a mature dataset.
"""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    import torch
    import yaml
    from ultralytics import YOLO
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/column-specialist-data')
    args=parser.parse_args()
    source=args.source.resolve()
    rows=json.loads((source/'review.json').read_text())
    destination=source/'dataset'
    if destination.exists():raise ValueError('Use a fresh output; never overwrite a study')
    def group(row):
        if row['scene'].startswith('vlc-record'):return 'same_cctv_camera_conservative_group'
        if row['scene'] in ['IMG_6874_MOV','IMG_6875_MOV']:return 'mall_steps_recordings_group'
        return row['scene']
    groups=defaultdict(list)
    for row in rows:groups[group(row)].append(row)
    positive_groups=sorted(g for g,items in groups.items() if any(r['positive'] for r in items))
    negative_groups=sorted(g for g,items in groups.items() if not any(r['positive'] for r in items))
    rng=random.Random(0);rng.shuffle(positive_groups);rng.shuffle(negative_groups)
    fixed=source/'frozen_heldout_groups.json'
    heldout=set(json.loads(fixed.read_text())) if fixed.exists() else set(
        positive_groups[:max(2,round(len(positive_groups)*.2))]+negative_groups[:2])
    counts=defaultdict(int)
    for row in rows:
        split='val' if group(row) in heldout else 'train'
        (destination/split/'images').mkdir(parents=True,exist_ok=True)
        (destination/split/'labels').mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/(row['id']+'.jpg'),destination/split/'images'/(row['id']+'.jpg'))
        text=''
        if row['positive']:
            x1,y1,x2,y2=row['label_box'];w,h=row['size']
            text=f'0 {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n'
        (destination/split/'labels'/(row['id']+'.txt')).write_text(text)
        row.update(split=split,source_group=group(row),reviewer='AI crop contact-sheet review; team review pending')
        counts[split]+=1
    (destination/'data.yaml').write_text(yaml.safe_dump(dict(path=str(destination),train='train/images',val='val/images',names=['structural column'],nc=1)))
    plan=dict(rows=rows,counts=dict(counts),seed=0,epochs=40,freeze=10,batch=4,imgsz=416,
        use='Experimental standalone specialist only; original five-class checkpoint unchanged',
        caveat=f'{len(rows)} selected crops; incomplete domain coverage. Recorded-source grouping cannot prove site independence.',
        new_sg_challenge_used_for_training=False,test_used=False)
    (source/'training-plan.json').write_text(json.dumps(plan,indent=2))
    model=YOLO(str(ROOT/'models/general/yolov8n.pt'))
    model.train(data=str(destination/'data.yaml'),epochs=40,patience=0,batch=4,imgsz=416,
        freeze=10,device='cpu',workers=0,seed=0,deterministic=True,optimizer='AdamW',lr0=.001,
        warmup_epochs=1,warmup_bias_lr=.001,cos_lr=True,lrf=.1,
        mosaic=1.,close_mosaic=5,scale=.4,translate=.1,plots=False,
        project=str(source),name='pilot',save_period=-1)
    print('COLUMN_PILOT_COMPLETE',flush=True)


if __name__=='__main__':main()
