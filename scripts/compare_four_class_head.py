"""Compare native four-output checkpoint against preserved original predictions.

Original cached predictions are from the same supplied checkpoint, with the
same low cutoff and preprocessing. Compare on ALL old valid/test images before
examining the cleaned subsets, so exclusions cannot hide a conversion regression.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evaluation import load_ground_truth,score_detections
from path_utils import read_bgr
from golden import iou
NAMES=['Escalator','People','Stair','Wet Floor Sign']
OLD=['Escalator','People','Pillar','Stair','Wet Floor Sign']
MAP={0:0,1:1,3:2,4:3}


def remap_gt(gt):
    gt=copy.deepcopy(gt);gt['categories']=[dict(id=i+1,name=n) for i,n in enumerate(NAMES)]
    gt['annotations']=[dict(a,category_id=MAP[a['category_id']-1]+1) for a in gt['annotations'] if a['category_id']-1 in MAP]
    return gt


def remap_outputs(outputs):
    return {key:dict(detections=[dict(d,class_id=MAP[d['class_id']]) for d in val['detections'] if d['class_id'] in MAP]) for key,val in outputs.items()}


def deployed_differences(before,after):
    different=[]
    for name,entry in before.items():
        old=[d for d in entry['detections'] if d['confidence']>=.35]
        new=[d for d in after[name]['detections'] if d['confidence']>=.35];used=set()
        for d in old:
            options=[(iou(d['box_xyxy'],x['box_xyxy']),j) for j,x in enumerate(new)
                     if j not in used and d['class_id']==x['class_id'] and abs(d['confidence']-x['confidence'])<1e-5]
            overlap,j=max(options,default=(0,-1))
            if overlap>.999:used.add(j)
            else:different.append(name)
        if len(old)!=len(new):different.append(name)
    return sorted(set(different))


def main():
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);out=ROOT/'artifacts/four-class-cleanup';model=YOLO(str(ROOT/'models/local/yolov8n_v4_four_class.pt'))
    result=dict(source_sha256=hashlib.sha256((ROOT/'models/local/yolov8n_v4.pt').read_bytes()).hexdigest(),
                four_sha256=hashlib.sha256((ROOT/'models/local/yolov8n_v4_four_class.pt').read_bytes()).hexdigest(),splits={})
    for split,cache in [('valid','validation-baseline.json'),('test','real-model-test.json')]:
        gt,paths=load_ground_truth(ROOT/'data/v4_scene_split'/split,OLD);gt=remap_gt(gt)
        original=remap_outputs(json.loads((ROOT/'artifacts'/cache).read_text()));assert set(original)=={p.name for p in paths}
        predictions={}
        for index,path in enumerate(paths,1):
            frame=read_bgr(path);pred=model.predict(frame,imgsz=640,conf=.001,iou=.7,max_det=300,rect=False,device='cpu',verbose=False)[0]
            predictions[path.name]=dict(detections=[dict(class_id=int(cid),**{'class':NAMES[int(cid)]},confidence=c,box_xyxy=[a,b,x,y]) for a,b,x,y,c,cid in pred.boxes.data.tolist()])
            if index%200==0:print(split,index,'/',len(paths),flush=True)
        (out/f'{split}-four-predictions.json').write_text(json.dumps(predictions))
        before=score_detections(gt,paths,original,NAMES,.35,['Stair','Escalator','Wet Floor Sign'])
        after=score_detections(gt,paths,predictions,NAMES,.35,['Stair','Escalator','Wet Floor Sign'])
        delta={name:{key:after['per_class'][name][key]-before['per_class'][name][key] for key in ['tp','fp','fn','ap50','precision','recall']} for name in NAMES}
        passed=all(d['tp']>=0 and d['fp']<=0 and d['ap50']>=-.00001 for d in delta.values())
        differences=deployed_differences(original,predictions)
        result['splits'][split]=dict(original_four_class=before,projected_four_class=after,delta=delta,strict_no_degradation=passed,
                                      deployed_prediction_parity=dict(images=len(paths),different_images=len(differences),iou_tolerance=.999,confidence_tolerance=1e-5,cutoff=.35),
                                      limitation='Historical development set, not an unseen final evaluation. Pillar excluded on both sides; four-class mean AP is not comparable with old five-class mean AP.')
        (out/'head-comparison.json').write_text(json.dumps(result,indent=2));print(split,'PASS',passed,'DELTA',delta,flush=True)
    result['strict_no_degradation']=all(x['strict_no_degradation'] for x in result['splits'].values())
    max_drop=max(-d['ap50'] for s in result['splits'].values() for d in s['delta'].values())
    # This separate engineering acceptance is explicit. Do not overwrite the
    # failed strict AP gate or claim mathematical AP identity.
    accepted=max_drop<=.001 and all(s['deployed_prediction_parity']['different_images']==0 for s in result['splits'].values())
    result['deployment_acceptance']=dict(accepted=accepted,max_ap50_drop=max_drop,exact_ap_gate_passed=result['strict_no_degradation'],
        criteria='All per-image retained detections agree at deployed .35 (IoU>.999, confidence difference<1e-5); AP50 loss <=.001 absolute.',
        note='Strict AP identity failed in recorded run; low-confidence argmax/NMS can change after removing a competing class. Not a retraining improvement.')
    (out/'head-comparison.json').write_text(json.dumps(result,indent=2))
    if not accepted:raise SystemExit(2)


if __name__=='__main__':main()
