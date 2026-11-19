"""Experimental pillar-only context padding; never enables itself in the app."""
from copy import deepcopy
import math


def context_input(frame,scale=.75):
    import cv2
    import numpy as np
    if not math.isfinite(scale) or not 0<scale<=1:raise ValueError('Invalid scale')
    height,width=frame.shape[:2]
    h,w=max(1,round(height*640/max(height,width))),max(1,round(width*640/max(height,width)))
    rh,rw=max(1,round(h*scale)),max(1,round(w*scale))
    top,left=(h-rh)//2,(w-rw)//2
    canvas=np.full((h,w,3),114,np.uint8)
    canvas[top:top+rh,left:left+rw]=cv2.resize(frame,(rw,rh))
    return canvas,dict(top=top,left=left,resized_height=rh,resized_width=rw,height=height,width=width)


def original_box(box,transform,min_retained_area=.8):
    t=transform
    x1,y1,x2,y2=box
    mapped=((x1-t['left'])*t['width']/t['resized_width'],
            (y1-t['top'])*t['height']/t['resized_height'],
            (x2-t['left'])*t['width']/t['resized_width'],
            (y2-t['top'])*t['height']/t['resized_height'])
    x1,y1,x2,y2=mapped
    if not all(math.isfinite(x) for x in mapped) or x2<=x1 or y2<=y1:return None
    clipped=(max(0.,x1),max(0.,y1),min(float(t['width']),x2),min(float(t['height']),y2))
    a,b,c,d=clipped
    if a>=c or b>=d or (c-a)*(d-b)<min_retained_area*(x2-x1)*(y2-y1):return None
    return list(clipped)


def merge_pillars(base,extra,threshold=.5):
    from golden import iou
    others=[deepcopy(d) for d in base if d['class_id']!=2]
    pillars=[deepcopy(d) for d in base if d['class_id']==2]
    pillars += [deepcopy(d) for d in extra if d['class_id']==2 and d['confidence']>=threshold]
    kept=[]
    for d in sorted(pillars,key=lambda x:-x['confidence']):
        if not any(iou(d['box_xyxy'],old['box_xyxy'])>=.7 for old in kept):kept.append(d)
    return sorted(others+kept,key=lambda x:-x['confidence'])
