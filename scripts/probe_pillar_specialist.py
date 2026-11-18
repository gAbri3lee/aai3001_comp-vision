"""Actual open-vocabulary Pillar smoke test, separate from trained-model metrics."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLOWorld
    from golden import iou
    from path_utils import read_bgr
    config = json.loads((ROOT/'configs/pillar_open_vocabulary.json').read_text())
    output = ROOT/'artifacts/pillar-open-vocabulary'
    output.mkdir(parents=True,exist_ok=True)
    # The text encoder's default weights directory is relative to this cwd.
    os.chdir(output)
    torch.set_num_threads(4)
    fixed = output/'pillar-worldv2-fixed.pt'
    if fixed.exists():
        model = YOLOWorld(str(fixed))
        if list(model.names.values()) != config['prompts']:
            raise ValueError('Cached vocabulary differs')
    else:
        model = YOLOWorld(str(output/config['model']))
        model.set_classes(config['prompts'])
        model.save(str(fixed))
        model = YOLOWorld(str(fixed))
    targets = {r['id']:r['target'] for r in json.loads((ROOT/'annotations/online_primary_targets.json').read_text())['records']}
    sources = json.loads((ROOT/'data/online_sg_malls/sources.json').read_text())
    sources = [r for r in sources if Path(r.get('file','')).suffix.lower() in ('.jpg','.jpeg','.png')]
    cases = [(r['id'],read_bgr(ROOT/'data/online_sg_malls'/r['file'])) for r in sources]
    cases += [('black blank',np.zeros((480,640,3),dtype=np.uint8)),
              ('white blank',np.full((480,640,3),255,dtype=np.uint8))]
    reports = []
    for name,frame in cases:
        started = time.perf_counter()
        result = model.predict(frame,imgsz=config['input_size'],conf=config['confidence'],
            iou=config['nms_iou'],agnostic_nms=True,rect=False,device='cpu',verbose=False)[0]
        elapsed = 1000*(time.perf_counter()-started)
        detections = [dict(prompt=config['prompts'][int(row[5])],confidence=float(row[4]),
                          box_xyxy=[float(v) for v in row[:4]])
            for row in result.boxes.data.cpu().tolist() if int(row[5]) in config['pillar_prompt_ids']]
        target = targets.get(name)
        best = max((iou(d['box_xyxy'],target['box_xyxy']) for d in detections),default=0) if target else None
        reports.append(dict(id=name,pillar_detections=detections,elapsed_ms=elapsed,
                            primary_target_iou=best,primary_target_seen=best>=.5 if best is not None else None))
        if target or detections:
            drawn = result.plot()
            if target:
                x1,y1,x2,y2 = map(int,target['box_xyxy'])
                cv2.rectangle(drawn,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.imwrite(str(output/(name+'_comparison.jpg')),drawn)
        print(json.dumps(reports[-1]),flush=True)
    report = dict(config=config,checkpoint_sha256=hashlib.sha256(fixed.read_bytes()).hexdigest(),
        checkpoint_bytes=fixed.stat().st_size,inputs=reports,
        note='Selected, incompletely annotated public images. Primary-target coverage only; no all-object accuracy claim.',
        application_changed=False)
    (output/'smoke.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
