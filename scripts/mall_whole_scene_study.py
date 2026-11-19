"""Matched full-image column pilots; venue-disjoint test remains sealed until training ends."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from path_utils import read_bgr


def prepare(manifest,out):
    import cv2
    import yaml
    rows=manifest['images']
    splits={}
    for r in rows:
        splits.setdefault(r['venue'],set()).add(r['split'])
        if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Image changed')
    if any(len(v)>1 for v in splits.values()):raise ValueError('Venue leakage')
    for arm in manifest['predeclared']['arms']:
        data=out/arm/'dataset'
        for split in ['train','val']:
            (data/split/'images').mkdir(parents=True)
            (data/split/'labels').mkdir(parents=True)
        for row in rows:
            if row['split']=='test':continue  # Never decode test pixels in preparation/training.
            image=read_bgr(ROOT/row['path']);h,w=image.shape[:2]
            labels=''.join(f'0 {(a+c)/(2*w):.6f} {(b+d)/(2*h):.6f} {(c-a)/w:.6f} {(d-b)/h:.6f}\n' for a,b,c,d in row['boxes'])
            for variant in range(4 if row['split']=='train' else 1):
                frame=image.copy()
                if arm=='degradation_augmented' and row['split']=='train':
                    if variant==1:frame=(frame*.35).astype('uint8')
                    elif variant==2:frame=cv2.GaussianBlur(frame,(21,21),0)
                    elif variant==3:frame=cv2.imdecode(cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,25])[1],cv2.IMREAD_COLOR)
                name=f"{row['id']}_{variant}"
                cv2.imwrite(str(data/row['split']/'images'/(name+'.jpg')),frame)
                (data/row['split']/'labels'/(name+'.txt')).write_text(labels)
        (data/'data.yaml').write_text(yaml.safe_dump(dict(path=str(data),train='train/images',val='val/images',nc=1,names=['large structural column'])))


def main():
    import torch
    from ultralytics import YOLO
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/mall-whole-scene-study')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError('Refuse overwrite; use a fresh study directory')
    manifest=json.loads((ROOT/'annotations/mall_whole_scene_columns.json').read_text())
    out.mkdir(parents=True);(out/'frozen_protocol.json').write_text(json.dumps(manifest,indent=2))
    prepare(manifest,out);torch.set_num_threads(4)
    status=dict(completed=[],test_evaluated=False)
    for seed in manifest['predeclared']['seeds']:
        for arm in manifest['predeclared']['arms']:
            print('START_RUN',arm,seed,flush=True)
            model=YOLO(str(ROOT/'models/general/yolov8n.pt'))
            model.train(data=str(out/arm/'dataset/data.yaml'),epochs=30,patience=0,batch=4,imgsz=416,
                freeze=10,device='cpu',workers=0,seed=seed,deterministic=True,optimizer='AdamW',lr0=.001,
                warmup_epochs=1,warmup_bias_lr=.001,cos_lr=True,lrf=.1,
                mosaic=1.,close_mosaic=5,scale=.4,translate=.1,plots=False,
                project=str(out/arm),name='seed'+str(seed),save_period=-1)
            last=out/arm/('seed'+str(seed))/'weights/last.pt'
            status['completed'].append(dict(arm=arm,seed=seed,last_sha256=hashlib.sha256(last.read_bytes()).hexdigest()))
            (out/'status.json').write_text(json.dumps(status,indent=2));del model
    print('ALL_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
