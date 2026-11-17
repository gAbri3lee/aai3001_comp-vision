"""Real native four-class checkpoint, HTTP uploads and legacy cloud contract.

Only absent camera acquisition is replaced; no fake detector predictions.
Physical speech, distance and hosted cloud are outside this isolated check.
"""
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from capture import LatestFrameCapture
from config import load_config
from golden import iou
from helpers import make_args
from verify_real_uploads import IdleCamera
from webapp import PipelineState,create_app


def main():
    import cv2
    import requests
    import torch
    from werkzeug.serving import make_server,WSGIRequestHandler
    torch.set_num_threads(4);config=load_config();names=config['classes']
    assert names==['Escalator','People','Stair','Wet Floor Sign']
    rows=json.loads((ROOT/'annotations/four_class_external.json').read_text())['images']
    previous=json.loads((ROOT/'artifacts/four-class-validation/report.json').read_text());prior={c['id']:c for c in previous['cases']}
    out=ROOT/'artifacts/four-class-cleanup';report=dict(classes=names,real_weights=True,camera=False,speech=False,depth=False,hosted_cloud=False,cases=[])
    def capture(source,**kwargs):return IdleCamera() if isinstance(source,int) else LatestFrameCapture(source,**kwargs)
    class Quiet(WSGIRequestHandler):
        def log_request(self,*a,**kw):pass
    with patch('webapp.LatestFrameCapture',side_effect=capture):
        state=PipelineState(make_args());server=make_server('127.0.0.1',0,create_app(state),threaded=True,request_handler=Quiet)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start();base=f'http://127.0.0.1:{server.server_port}';session=requests.Session()
        try:
            info=session.get(base+'/config',timeout=5).json();assert info['classes']==names and 'Pillar' not in info['priority']
            assert [x['class'] for x in session.get(base+'/demo',timeout=5).json()['classes']]==names
            report['UI_config_and_library_four_classes']=True
            response=session.post(base+'/analyze',data={'expected':'Pillar'},files={'image':('image.jpg',(ROOT/rows[0]['path']).read_bytes(),'image/jpeg')},timeout=10)
            assert response.status_code==400;report['pillar_expected_rejected']=True
            files=[('image',(r['id']+'.jpg',(ROOT/r['path']).read_bytes(),'image/jpeg')) for r in rows]+[('image',('corrupt.jpg',b'bad','image/jpeg'))]
            response=session.post(base+'/analyze',files=files,data={'view':'recognition'},timeout=90);response.raise_for_status();results=response.json()['results'];assert len(results)==len(rows)+1 and 'error' in results[-1]
            report['corrupt_image_isolated']=True
            for row,result in zip(rows,results):
                run=result['runs'][0];assert 'error' not in run
                ds=run['detections'];assert all(d['class']!='Pillar' for d in ds)
                assert all(d['class_id']==-1 or names[d['class_id']]==d['class'] for d in ds)
                h=cv2.imread(str(ROOT/row['path'])).shape[0]
                trained=[d for d in ds if d['class_id']>=0 and d['box_xyxy'][3]-d['box_xyxy'][1]>=h*.1]
                # Same original ignore regions, matched by class names after ID migration.
                old_names=['Escalator','People','Pillar','Stair','Wet Floor Sign']
                trained=[d for d in trained if not any(d['class'] in [old_names[c] for c in region['classes']] and iou(d['box_xyxy'],region['box_xyxy'])>=.5 for region in row.get('ignore',[]))]
                a=prior[row['id']]['trained'];used=set()
                for d in a:
                    options=[(iou(d['box_xyxy'],x['box_xyxy']),j) for j,x in enumerate(trained) if j not in used and d['class']==x['class'] and abs(d['confidence']-x['confidence'])<1e-5]
                    overlap,j=max(options,default=(0,-1));assert overlap>.999,(row['id'],d);used.add(j)
                assert len(a)==len(trained),(row['id'],len(a),len(trained))
                report['cases'].append(dict(id=row['id'],trained_predictions_match_previous=True,detections=ds,recognition=run['recognition'],alert=run['alert']))
            videos=[ROOT/'data/four_class_videos/opencv-vtest-first20s.mp4',ROOT/'artifacts/online-source-uploads-isolated-final/vivocity-source-converted.mp4'];report['videos']=[]
            for video in videos:
                response=session.post(base+'/source',files={'video':(video.name,video.read_bytes(),'video/mp4')},timeout=20);response.raise_for_status();start=time.monotonic()
                while time.monotonic()-start<45:
                    status=session.get(base+'/status',timeout=5).json()
                    assert all(d['class']!='Pillar' and (d['class_id']==-1 or names[d['class_id']]==d['class']) for d in status['detections'])
                    if status['source']['ended']:break
                    time.sleep(.2)
                assert status['source']['ended'] and status['frames_processed']>0
                assert [x['class'] for x in status['video_summary']]==names
                report['videos'].append(dict(file=video.name,frames=status['frames_processed'],summary=status['video_summary'],completed=True))
            assert session.post(base+'/source/control',json={'action':'restart'},timeout=5).status_code==200
            assert session.post(base+'/source/control',json={'action':'camera'},timeout=5).status_code==200
            report['replay_and_return_to_camera']=True
        finally:
            server.shutdown();worker.join(timeout=5);server.server_close();state.stop();session.close()
    report['cleanup_passed']=state.store.root is None;assert report['cleanup_passed']
    (out/'runtime-verification.json').write_text(json.dumps(report,indent=2));print('PASS: 11 real photo uploads, two real videos, four-class UI config, no pillar outputs, corrupted-file isolation and cleanup.',flush=True)


if __name__=='__main__':main()
