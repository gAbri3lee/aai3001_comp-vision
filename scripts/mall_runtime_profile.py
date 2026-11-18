"""Temporary research supplement for profiling only. Does not modify app defaults."""
from dataclasses import replace
import json,sys,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from detection import Detection
from golden import iou


class ColumnSupplement:
    def __init__(self,settings,weights,**kwargs):
        from scene_detector import SceneDetector
        from ultralytics import YOLO
        self.base=SceneDetector(settings,**kwargs);self.column=YOLO(str(weights))
    def __getattr__(self,name):return getattr(self.base,name)
    def predict(self,frame):
        batch=self.base.predict(frame);start=time.perf_counter()
        result=self.column.predict(frame,imgsz=416,conf=.5,iou=.7,rect=False,device='cpu',verbose=False)[0]
        extra=[]
        for x1,y1,x2,y2,c,k in result.boxes.data.tolist():
            box=(x1,y1,x2,y2)
            if y2-y1<frame.shape[0]*.15 or x2-x1<frame.shape[1]*.02:continue
            if any(d.label=='Pillar' and iou(d.box_xyxy,box)>=.5 for d in batch.detections):continue
            extra.append(Detection(2,'Pillar',c,box))
        return replace(batch,detections=tuple(sorted(batch.detections+tuple(extra),key=lambda d:-d.confidence)),
            latency_ms=batch.latency_ms+1000*(time.perf_counter()-start),model=batch.model+' + RESEARCH column')


def main():
    import benchmark_app
    study=ROOT/'artifacts/mall-whole-scene-study';evaluation=json.loads((study/'evaluation.json').read_text())
    weights=study/evaluation['selected_arm']/'seed0/weights/last.pt'
    with patch('webapp.create_detector',side_effect=lambda settings,**kw:ColumnSupplement(settings,weights,**kw)):
        benchmark_app.main()


if __name__=='__main__':main()
