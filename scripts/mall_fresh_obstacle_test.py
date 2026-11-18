"""Fresh, type-scoped object benchmark; not unknown-obstacle accuracy."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from golden import iou
from path_utils import read_bgr


def main():
    import torch
    from ultralytics import YOLO,YOLOWorld
    torch.set_num_threads(4);out=ROOT/'artifacts/mall-whole-scene-study/fresh-obstacles.json'
    if out.exists():raise ValueError('Preserve the original fresh-test result')
    if not json.loads((out.parent/'status.json').read_text())['test_evaluated']:raise RuntimeError('Finish column study first')
    rows=json.loads((ROOT/'annotations/mall_fresh_obstacles.json').read_text())['images'];reports={}
    for name,model,mapping in [('COCO',YOLO(str(ROOT/'models/general/yolov8n.pt')),{13:'seating',56:'seating',57:'seating',58:'planter'}),
        ('World',YOLOWorld(str(ROOT/'artifacts/mall-world-obstacles/mall-world.pt')),{1:'seating',2:'seating',3:'bin',4:'planter'})]:
        cases=[]
        for row in rows:
            frame=read_bgr(ROOT/row['path']);start=time.perf_counter()
            result=model.predict(frame,imgsz=640,conf=.35,rect=False,device='cpu',verbose=False)[0];ds=[]
            for x1,y1,x2,y2,c,k in sorted(result.boxes.data.tolist(),key=lambda x:-x[4]):
                if int(k) not in mapping or y2-y1<frame.shape[0]*.1:continue
                d=dict(label=mapping[int(k)],box=[x1,y1,x2,y2],confidence=c)
                if not any(p['label']==d['label'] and iou(p['box'],d['box'])>=.5 for p in ds):ds.append(d)
            cases.append(dict(id=row['id'],targets=row['objects'],predictions=ds,elapsed_ms=1000*(time.perf_counter()-start)))
        metrics={}
        for threshold in [.35,.5]:
            tp=fp=fn=0
            for row in cases:
                used=set()
                for d in row['predictions']:
                    if d['confidence']<threshold:continue
                    candidates=[(iou(d['box'],t['box_xyxy']),i) for i,t in enumerate(row['targets']) if i not in used and t['label']==d['label']]
                    overlap,index=max(candidates,default=(0,None))
                    if overlap>=.5:tp+=1;used.add(index)
                    else:fp+=1
                fn+=len(row['targets'])-len(used)
            metrics[str(threshold)]=dict(tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn))
        reports[name]=dict(metrics=metrics,cases=cases)
    out.write_text(json.dumps(dict(scope='Bins, seating and planters >=10% height only; two new venues; AI labels need independent review',results=reports),indent=2));print({k:v['metrics'] for k,v in reports.items()},flush=True)


if __name__=='__main__':main()
