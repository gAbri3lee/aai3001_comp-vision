"""One final evaluation of six fixed-epoch models; no test-based selection."""
import hashlib,json
from pathlib import Path
import sys
import statistics
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from path_utils import read_bgr
from evaluation import score_detections


def main():
    import torch
    from ultralytics import YOLO
    out=ROOT/'artifacts/mall-whole-scene-study';status=json.loads((out/'status.json').read_text())
    if len(status['completed'])!=9:raise RuntimeError('All nine matched runs must complete before test')
    if status['test_evaluated']:raise RuntimeError('Test already evaluated; preserve first-run evidence')
    manifest=json.loads((out/'frozen_protocol.json').read_text());torch.set_num_threads(4)
    # F001's storefront support is ambiguous under the freestanding-column rule.
    # Excluded before any predictions, independently of model results.
    (out/'preprediction_addendum.json').write_text(json.dumps(dict(excluded_test_ids=['F001'],reason='Ambiguous storefront wall pier versus freestanding support; AI review cannot resolve reliably',selection='mean validation F1 across three seeds; seed0 for runtime profiling'),indent=2))
    rows=[r for r in manifest['images'] if r['split'] in ['val','test'] and r['id']!='F001']
    def evaluate(weights,baseline=False):
        model=YOLO(str(weights));outputs={}
        for row in rows:
            result=model.predict(read_bgr(ROOT/row['path']),imgsz=640 if baseline else 416,
                conf=.001,rect=False,iou=.7,device='cpu',verbose=False)[0]
            h,w=result.orig_shape;ds=[]
            for x1,y1,x2,y2,c,k in result.boxes.data.tolist():
                if baseline and int(k)!=2:continue
                if y2-y1<h*.15 or x2-x1<w*.02:continue
                ds.append(dict(class_id=0,**{'class':'large structural column'},confidence=c,box_xyxy=[x1,y1,x2,y2]))
            outputs[Path(row['path']).name]=dict(detections=ds)
        metrics={}
        for split in ['val','test']:
            subset=[r for r in rows if r['split']==split];images=[];anns=[];paths=[]
            from PIL import Image
            for i,row in enumerate(subset):
                path=ROOT/row['path'];w,h=Image.open(path).size;paths.append(path)
                images.append(dict(id=i,file_name=path.name,width=w,height=h))
                for a,b,c,d in row['boxes']:anns.append(dict(id=len(anns)+1,image_id=i,category_id=1,bbox=[a,b,c-a,d-b],area=(c-a)*(d-b),iscrowd=0))
            gt=dict(images=images,annotations=anns,categories=[dict(id=1,name='large structural column')])
            metrics[split]=score_detections(gt,paths,{p.name:outputs[p.name] for p in paths},['large structural column'],.35 if baseline else .5,[])
        return metrics,outputs
    report=dict(protocol=manifest['predeclared'],excluded_test_ids=['F001'],runs=[],baseline=None,
        limitation='Small AI-labelled pilot: two validation scenes and four test scenes from two unseen venues; not a statistically strong or human-certified benchmark.')
    report['baseline'],base=evaluate(ROOT/'models/local/yolov8n_v4.pt',True)
    for run in status['completed']:
        weights=out/run['arm']/('seed'+str(run['seed']))/'weights/last.pt'
        assert hashlib.sha256(weights.read_bytes()).hexdigest()==run['last_sha256']
        metrics,pred=evaluate(weights);record=dict(run,metrics=metrics);report['runs'].append(record)
        (out/(run['arm']+'_seed'+str(run['seed'])+'_predictions.json')).write_text(json.dumps(pred))
        print('EVALUATED',run['arm'],run['seed'],metrics['test']['per_class'],flush=True)
    report['validation_mean_f1']={arm:statistics.mean(r['metrics']['val']['f1'] for r in report['runs'] if r['arm']==arm) for arm in manifest['predeclared']['arms']}
    report['selected_arm']=max(report['validation_mean_f1'],key=report['validation_mean_f1'].get)
    report['selected_seed']=0
    (out/'evaluation.json').write_text(json.dumps(report,indent=2));status['test_evaluated']=True;(out/'status.json').write_text(json.dumps(status,indent=2))
    print('EVALUATION_COMPLETE',report['selected_arm'],flush=True)


if __name__=='__main__':main()
