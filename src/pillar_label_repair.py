"""Apply explicitly reviewed TRAIN label edits with immutable-source checks."""
import hashlib
import math


def repair_label(original: bytes, decision: dict, width: int, height: int):
    if decision.get('reviewed_full_image') is not True or not decision.get('reviewer'):
        raise ValueError('Full-image visual adjudication and reviewer attribution required')
    raw_hash=hashlib.sha256(original).hexdigest()
    canonical_hash=hashlib.sha256(original.replace(b'\r\n',b'\n')).hexdigest()
    if (raw_hash != decision.get('original_label_sha256') and
            canonical_hash != decision.get('canonical_label_sha256')):
        raise ValueError('Original label changed; repeat review')
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    lines = original.decode('utf-8').splitlines()
    for line in decision.get('remove_lines',[]):
        if line not in lines:
            raise ValueError('Reviewed removal does not exist')
        lines.remove(line)
    additions=[]
    for box in decision.get('add_boxes',[]):
        cid = box['class_id']
        if isinstance(cid,bool) or not isinstance(cid,int) or cid not in range(5):
            raise ValueError('Class ID must remain canonical')
        coords = box['xyxy']
        if len(coords)!=4 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in coords):
            raise ValueError('Invalid reviewed box')
        x1,y1,x2,y2=coords
        if not (0<=x1<x2<=width and 0<=y1<y2<=height):
            raise ValueError('Reviewed box outside image')
        line=f'{cid} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}'
        if line in lines:
            raise ValueError('Duplicate reviewed annotation')
        lines.append(line)
        additions.append(line)
    return ('\n'.join(lines)+'\n' if lines else '').encode(), additions
