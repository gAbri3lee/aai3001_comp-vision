"""Check validation-selected supplement against all 925 unchanged v4 validation images."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from path_utils import read_bgr
from golden import iou
from evaluation import load_ground_truth,score_detections
from config import load_config


def main():
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);out=ROOT/'artifacts/mall-whole-scene-study';e=json.loads((out/'evaluation.json').read_text());weights=out/e['selected_arm']/'seed0/weights/last.pt';model=YOLO(str(weights))
    baseline=json.loads((ROOT/'artifacts/validation-baseline.json').read_text());cfg=load_config();gt,paths=load_ground_truth(ROOT/'data/v4_scene_split/valid',cfg['classes']);outputs={};times=[]
    for n,path in enumerate(paths):
        frame=read_bgr(path);start=time.perf_counter();r=model.predict(frame,imgsz=416,conf=.5,iou=.7,rect=False,device='cpu',verbose=False)[0];times.append(1000*(time.perf_counter()-start));old=baseline[path.name]['detections'];extra=[]
        for x1,y1,x2,y2,c,k in r.boxes.data.tolist():
            box=[x1,y1,x2,y2]
            if y2-y1<frame.shape[0]*.15 or x2-x1<frame.shape[1]*.02:continue
            if any(d['class_id']==2 and d['confidence']>=.35 and iou(d['box_xyxy'],box)>=.5 for d in old):continue
            extra.append(dict(class_id=2,**{'class':'Pillar'},confidence=c,box_xyxy=box))
        kept=[d for d in old if not (d['class_id']==2 and d['confidence']<.35 and any(iou(d['box_xyxy'],a['box_xyxy'])>=.5 for a in extra))]
        outputs[path.name]=dict(baseline[path.name],detections=sorted(kept+extra,key=lambda d:-d['confidence']))
        if (n+1)%200==0:print('VALIDATION',n+1,flush=True)
    before=score_detections(gt,paths,baseline,cfg['classes'],.35,cfg['safety_classes']);after=score_detections(gt,paths,outputs,cfg['classes'],.35,cfg['safety_classes']);b,a=before['per_class']['Pillar'],after['per_class']['Pillar']
    passed=a['recall']>=b['recall']+.1 and a['precision']>=max(.75,b['precision']-.02) and after['map50']>=before['map50']-.01
    (out/'supplement-validation.json').write_text(json.dumps(dict(before=before,after=after,gate_passed=passed,extra_mean_ms=sum(times)/len(times),application_changed=False),indent=2));print('GATE',passed,a,flush=True)


if __name__=='__main__':main()
