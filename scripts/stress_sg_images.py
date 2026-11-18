"""Controlled degradations of real mall images; not physical distance tests."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scene_detector import SceneDetector
from config import add_detector_arguments,detector_settings
from validate_sg_indoor import match_targets
from path_utils import read_bgr
from decision_engine import assess


def main():
    import cv2
    import numpy as np
    import torch
    parser=argparse.ArgumentParser();add_detector_arguments(parser)
    settings=detector_settings(parser.parse_args([]));model=SceneDetector(settings);torch.set_num_threads(4)
    manifest=json.loads((ROOT/'annotations/sg_indoor_primary_targets.json').read_text());rows=[]
    for row in manifest['images']:
        if row['id'] not in ['M001','M009','M010','M011','M012','M017']:continue
        original=read_bgr(ROOT/row['path'])
        for condition in ['normal','dim','blur','jpeg25','low_resolution']:
            image=original.copy();targets=row['targets']
            if condition=='dim':image=(image.astype(np.float32)*.35).astype(np.uint8)
            elif condition=='blur':image=cv2.GaussianBlur(image,(21,21),0)
            elif condition=='jpeg25':image=cv2.imdecode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,25])[1],cv2.IMREAD_COLOR)
            elif condition=='low_resolution':
                scale=320/max(image.shape[:2]);image=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
                targets=[dict(t,box_xyxy=[v*(image.shape[1]/original.shape[1] if i%2==0 else image.shape[0]/original.shape[0])
                    for i,v in enumerate(t['box_xyxy'])]) for t in targets]
            batch=model.predict(image);matches=match_targets(targets,batch.detections)
            hazards=assess(batch.detections,(image.shape[1],image.shape[0]),settings['alert_gate'],settings['severity'])
            rows.append(dict(id=row['id'],condition=condition,
                targets=[dict(label=t['label'],matched=i in matches) for i,t in enumerate(targets)],
                gated_alerts=[h.to_dict() for h in hazards if h.gated]))
    report=dict(conditions='Controlled image degradations, not measured walking/camera conditions or metres',inputs=len(rows),rows=rows)
    (ROOT/'artifacts/sg-indoor-validation/robustness.json').write_text(json.dumps(report,indent=2))
    print('ROBUSTNESS_DONE',len(rows),flush=True)


if __name__=='__main__':main()
