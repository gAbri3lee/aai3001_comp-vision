"""Additional four-class photo/key-frame validation and real HTTP upload checks.

Fixed deployed weights/thresholds. No training, scope hiding or class-ID changes.
"""
import hashlib,io,json,sys,time,threading
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from config import load_config
from detection import Detection
from decision_engine import assess
from evaluation import score_detections
from golden import iou
from path_utils import read_bgr
from helpers import make_args
from verify_real_uploads import IdleCamera
from capture import LatestFrameCapture
from webapp import PipelineState,create_app


def valid_prediction(d,row,shape):
    if d['class_id'] not in [0,1,3,4]:return False
    if d['box_xyxy'][3]-d['box_xyxy'][1]<shape[0]*.1:return False
    return not any(d['class_id'] in region['classes'] and iou(d['box_xyxy'],region['box_xyxy'])>=.5 for region in row.get('ignore',[]))


def main():
    import cv2,numpy as np,torch,requests
    from ultralytics import YOLO
    from werkzeug.serving import make_server,WSGIRequestHandler
    torch.set_num_threads(4);out=ROOT/'artifacts/four-class-validation';out.mkdir(exist_ok=True)
    # This report deliberately evaluates the original five-output checkpoint.
    # Pin its historical schema after the active application migrates to four.
    legacy=ROOT/'configs/legacy_v4_five_class.json'
    cfg=load_config(legacy);manifest=json.loads((ROOT/'annotations/four_class_external.json').read_text());rows=manifest['images']
    if (out/'report.json').exists() and not (out/'report-before-annotation-review.json').exists():
        (out/'report-before-annotation-review.json').write_bytes((out/'report.json').read_bytes())
    (out/'frozen_annotations.json').write_text(json.dumps(manifest,indent=2))
    model=YOLO(str(ROOT/'models/local/yolov8n_v4.pt'));predictions={};cases=[];gt=dict(images=[],annotations=[],categories=[dict(id=i+1,name=n) for i,n in enumerate(cfg['classes'])])
    for index,row in enumerate(rows):
        path=ROOT/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        frame=read_bgr(path);h,w=frame.shape[:2];result=model.predict(frame,imgsz=640,conf=.001,iou=.7,rect=False,device='cpu',verbose=False)[0]
        ds=[dict(class_id=int(k),**{'class':cfg['classes'][int(k)]},confidence=c,box_xyxy=[a,b,x,y]) for a,b,x,y,c,k in result.boxes.data.tolist()]
        ds=[d for d in ds if valid_prediction(d,row,frame.shape)]
        predictions[path.name]=dict(detections=ds);gt['images'].append(dict(id=index,file_name=path.name,width=w,height=h))
        for t in row['targets']:
            a,b,x,y=t['box_xyxy'];gt['annotations'].append(dict(id=len(gt['annotations'])+1,image_id=index,category_id=t['class_id']+1,bbox=[a,b,x-a,y-b],area=(x-a)*(y-b),iscrowd=0))
        high=[d for d in ds if d['confidence']>=.35];case=dict(id=row['id'],venue=row['venue'],targets=row['targets'],trained=high)
        for d in high:
            a,b,x,y=map(int,d['box_xyxy']);cv2.rectangle(frame,(a,b),(x,y),(0,165,255),2);cv2.putText(frame,f"{d['class']} {d['confidence']:.2f}",(a,max(16,b)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,165,255),2)
        for t in row['targets']:cv2.rectangle(frame,tuple(map(int,t['box_xyxy'][:2])),tuple(map(int,t['box_xyxy'][2:])),(0,255,0),2)
        cv2.imwrite(str(out/(row['id']+'.jpg')),frame);cases.append(case)
    metrics=score_detections(gt,[ROOT/r['path'] for r in rows],predictions,cfg['classes'],.35,cfg['safety_classes'])
    # Every remaining class has positive GT; COCO excludes the absent Pillar category.
    report=dict(scope=manifest['scope'],review=manifest['review'],metrics=metrics,cases=cases,
        weights_sha256=hashlib.sha256((ROOT/'models/local/yolov8n_v4.pt').read_bytes()).hexdigest(),
        trained_only_AP=True,application_changed=False,camera=False,speech=False,cloud=False,
        annotation_revision=manifest.get('annotation_revision'))
    def capture(source,**kwargs):return IdleCamera() if isinstance(source,int) else LatestFrameCapture(source,**kwargs)
    class Quiet(WSGIRequestHandler):
        def log_request(self,*args,**kwargs):pass
    with patch('webapp.LatestFrameCapture',side_effect=capture):
        state=PipelineState(make_args('--config',str(legacy)));server=make_server('127.0.0.1',0,create_app(state),threaded=True,request_handler=Quiet);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}';session=requests.Session()
        try:
            files=[('image',(r['id']+'.jpg',(ROOT/r['path']).read_bytes(),'image/jpeg')) for r in rows]+[('image',('broken.jpg',b'broken','image/jpeg'))]
            response=session.post(base+'/analyze',files=files,data={'view':'recognition'},timeout=90);response.raise_for_status();entries=response.json()['results']
            assert len(entries)==len(rows)+1, 'Every upload must produce a result'
            assert 'error' in entries[-1]
            for row,case,entry in zip(rows,cases,entries):
                run=entry['runs'][0];assert 'error' not in run
                frame=read_bgr(ROOT/row['path']);combined=[]
                for d in run['detections']:
                    item=dict(d)
                    if d['class_id']==-1 and d['class']=='People':item['class_id']=1
                    if valid_prediction(item,row,frame.shape):combined.append(item)
                case.update(combined=combined,actual_app_alert=run['alert'],recognition=run['recognition'])
            combined_outputs={Path(r['path']).name:dict(detections=c['combined']) for r,c in zip(rows,cases)}
            combined=score_detections(gt,[ROOT/r['path'] for r in rows],combined_outputs,cfg['classes'],.35,cfg['safety_classes'])
            report['combined_operating_counts']={name:{k:v for k,v in values.items() if k!='ap50'} for name,values in combined['per_class'].items()}
            report['combined_AP_not_reported']='General predictions use a fixed .35 cutoff; not low-threshold AP inputs.'
            video=ROOT/'data/four_class_videos/opencv-vtest-first20s.mp4'
            response=session.post(base+'/source',files={'video':('people20s.mp4',video.read_bytes(),'video/mp4')},timeout=20);response.raise_for_status();start=time.monotonic();samples=[]
            while time.monotonic()-start<45:
                status=session.get(base+'/status',timeout=5).json()
                if status['detections']:samples.append(dict(at=status['source']['position'],detections=status['detections'],alert=status['alert']))
                if status['source']['ended']:break
                time.sleep(.15)
            assert status['source']['ended'] and status['frames_processed']>0
            report['video']=dict(source='OpenCV vtest; outdoor people-motion supplement, not mall footage',duration=20,frames_processed=status['frames_processed'],summary=status['video_summary'],observations=samples)
            # Quantitative localisation on real decoded video frames. Each frame
            # uses the actual upload route, original weights and general model.
            keyframes=json.loads((ROOT/'annotations/four_class_video_frames.json').read_text())
            frame_rows=keyframes['images'];frame_gt=dict(images=[],annotations=[],categories=gt['categories']);frame_outputs={};frame_cases=[]
            for index,row in enumerate(frame_rows):
                path=ROOT/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
                frame=read_bgr(path);h,w=frame.shape[:2]
                frame_gt['images'].append(dict(id=index,file_name=path.name,width=w,height=h))
                for t in row['targets']:
                    a,b,x,y=t['box_xyxy'];frame_gt['annotations'].append(dict(id=len(frame_gt['annotations'])+1,image_id=index,category_id=t['class_id']+1,bbox=[a,b,x-a,y-b],area=(x-a)*(y-b),iscrowd=0))
                response=session.post(base+'/analyze',files={'image':(path.name,path.read_bytes(),'image/png')},timeout=30);response.raise_for_status()
                run=response.json()['results'][0]['runs'][0];assert 'error' not in run
                ds=[]
                for detection in run['detections']:
                    d=dict(detection)
                    if d['class_id']==-1 and d['class']=='People':d['class_id']=1
                    if valid_prediction(d,row,frame.shape):ds.append(d)
                frame_outputs[path.name]=dict(detections=ds)
                frame_cases.append(dict(at_seconds=row['at_seconds'],targets=row['targets'],detections=ds,alert=run['alert']))
                for d in ds:
                    a,b,x,y=map(int,d['box_xyxy']);cv2.rectangle(frame,(a,b),(x,y),(0,165,255),2)
                    cv2.putText(frame,f"{d['class']} {d['confidence']:.2f}",(a,max(16,b)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,165,255),1)
                for t in row['targets']:cv2.rectangle(frame,tuple(map(int,t['box_xyxy'][:2])),tuple(map(int,t['box_xyxy'][2:])),(0,255,0),2)
                cv2.imwrite(str(out/f"video-{row['at_seconds']}s.jpg"),frame)
            frame_metrics=score_detections(frame_gt,[ROOT/r['path'] for r in frame_rows],frame_outputs,cfg['classes'],.35,cfg['safety_classes'])
            report['video_keyframes']=dict(source=keyframes['source'],review=keyframes['review'],cases=frame_cases,
                operating_counts={name:{k:v for k,v in values.items() if k!='ap50'} for name,values in frame_metrics['per_class'].items()},
                limitation='Four related frames, not independent people or clips. Counts at fixed deployed cutoff; no video AP claim.')
            # Repeat the existing real indoor mall clip; clearly marked as reused.
            indoor=ROOT/'artifacts/online-source-uploads-isolated-final/vivocity-source-converted.mp4'
            response=session.post(base+'/source',files={'video':('vivocity.mp4',indoor.read_bytes(),'video/mp4')},timeout=20);response.raise_for_status();start=time.monotonic();indoor_samples=[]
            while time.monotonic()-start<30:
                status=session.get(base+'/status',timeout=5).json()
                if status['detections']:indoor_samples.append(dict(at=status['source']['position'],detections=status['detections'],alert=status['alert']))
                if status['source']['ended']:break
                time.sleep(.15)
            assert status['source']['ended'] and status['frames_processed']>0
            report['indoor_video_repeat']=dict(source='Previously used VivoCity escalator clip; NOT a new independent video',duration=status['source']['duration'],frames_processed=status['frames_processed'],summary=status['video_summary'],observations=indoor_samples)
            report['image_upload_passed']=True;report['corrupt_isolated']=True
        finally:server.shutdown();thread.join(timeout=5);server.server_close();state.stop();session.close()
    report['cleanup']=state.store.root is None
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'trained_predictions.json').write_text(json.dumps(predictions))
    print('FOUR_CLASSES',metrics['per_class'],flush=True);print('COMBINED',report['combined_operating_counts'],flush=True);print('VIDEO_FRAMES',report['video']['frames_processed'],flush=True)


if __name__=='__main__':main()
