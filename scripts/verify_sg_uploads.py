"""Real TCP uploads and actual weights on sourced Singapore mall media."""
import base64
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from capture import LatestFrameCapture
from helpers import make_args
from verify_real_uploads import IdleCamera
from webapp import PipelineState, create_app


def main():
    import cv2
    import requests
    import torch
    from werkzeug.serving import make_server
    torch.set_num_threads(4)
    out=ROOT/'artifacts/sg-indoor-validation/http';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'annotations/sg_indoor_primary_targets.json').read_text())
    rows=manifest['images']
    baseline={r['id']:r for r in json.loads((out.parent/'baseline/report.json').read_text())['images']}
    report=dict(camera=False,speech=False,cloud=False,actual_model=True,actual_tcp_http=True,
        browser_clickthrough=False,images=[],videos=[])
    def capture(source,**kwargs):
        return IdleCamera() if isinstance(source,int) else LatestFrameCapture(source,**kwargs)
    with patch('webapp.LatestFrameCapture',side_effect=capture):
        state=PipelineState(make_args())
        server=make_server('127.0.0.1',0,create_app(state),threaded=True)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        session=requests.Session();base=f'http://127.0.0.1:{server.server_port}'
        try:
            assert session.get(base,timeout=5).status_code==200
            for start in range(0,len(rows),19):
                subset=rows[start:start+19]
                files=[('image',(r['id']+'.jpg',(ROOT/r['path']).read_bytes(),'image/jpeg')) for r in subset]
                files.append(('image',('corrupt.jpg',b'broken image','image/jpeg')))
                response=session.post(base+'/analyze',files=files,data={'view':'alert','debug':'1'},timeout=90)
                response.raise_for_status();entries=response.json()['results'];assert 'error' in entries[-1]
                for row,entry in zip(subset,entries):
                    run=entry['runs'][0]
                    assert 'error' not in run
                    raw=run['images']['alert'].split(',',1)[1];(out/(row['id']+'.jpg')).write_bytes(base64.b64decode(raw))
                    expected=baseline[row['id']]['detections']
                    assert len(run['detections'])==len(expected),(row['id'],'HTTP/direct count mismatch')
                    for a,b in zip(run['detections'],expected):
                        assert a['class']==b['class'] and abs(a['confidence']-b['confidence'])<.001
                    report['images'].append(dict(id=row['id'],alert=run['alert'],
                        detections=run['detections'],recognition=run['recognition'],direct_parity=True))
            report['corrupt_images_isolated']=True
            response=session.post(base+'/source',files={'video':('corrupt.mp4',b'broken video','video/mp4')},timeout=10)
            assert response.status_code in (400,415,422)
            report['corrupt_video_rejected']=True
            # A real 9.47s mall clip already acquired, not a generated slideshow.
            video=ROOT/'artifacts/online-source-uploads-isolated-final/vivocity-source-converted.mp4'
            response=session.post(base+'/source',files={'video':('vivocity.mp4',video.read_bytes(),'video/mp4')},timeout=15)
            response.raise_for_status()
            response=session.post(base+'/source/control',json={'view':'alert'},timeout=5);response.raise_for_status()
            started=time.monotonic();samples=[];pause_tested=False;stream_tested=False
            while time.monotonic()-started<35:
                status=session.get(base+'/status',timeout=5).json()
                samples.append(status)
                if status['frames_processed'] and not pause_tested:
                    session.post(base+'/control',json={'paused':True},timeout=5).raise_for_status();time.sleep(.15)
                    n=session.get(base+'/status',timeout=5).json()['frames_processed'];time.sleep(.3)
                    assert session.get(base+'/status',timeout=5).json()['frames_processed']==n
                    session.post(base+'/control',json={'paused':False},timeout=5).raise_for_status();pause_tested=True
                if status['frames_processed'] and not stream_tested:
                    with session.get(base+'/video_feed',stream=True,timeout=5) as stream:
                        assert b'--frame' in next(stream.iter_content(1024))
                    stream_tested=True
                if status['source']['ended']:break
                time.sleep(.1)
            assert status['source']['ended'] and status['frames_processed']>0
            report['videos'].append(dict(source='VivoCity real Commons escalator clip; previously evaluated',
                ended=True,frames_processed=status['frames_processed'],summary=status['video_summary'],
                pause_resume= pause_tested,mjpeg=stream_tested,
                alert_samples=[dict(alert=s['alert'],detections=s['detections'],latency_ms=s['pipeline_latency_ms'])
                               for s in samples if s['detections']]))
            session.post(base+'/source/control',json={'action':'restart'},timeout=10).raise_for_status()
            report['replay_accepted']=True
            session.post(base+'/source/control',json={'action':'camera'},timeout=5).raise_for_status()
            report['return_to_camera_accepted']=True
        finally:
            server.shutdown();thread.join(timeout=5);server.server_close();state.stop();session.close()
    report['temporary_files_cleaned']=state.store.root is None
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(images=len(report['images']),
        processed_video_frames=[v['frames_processed'] for v in report['videos']],
        cleanup=report['temporary_files_cleaned'])),flush=True)


if __name__=='__main__':main()
