"""Apply only hash-verified, reviewed TRAIN-label additions; archive originals.

Control-arm records were frozen before these edits. Validation/test labels and
images are never changed. Every entry is checked before the first write.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from training_dataset import validate_dataset


def label_text(rows):
    lines=[]
    for cid,cx,cy,w,h in rows:
        if isinstance(cid,bool) or cid!=int(cid) or not 0<=cid<4:raise ValueError('Invalid four-class ID')
        if not all(math.isfinite(v) for v in [cx,cy,w,h]) or not (0<w<=1 and 0<h<=1):raise ValueError('Invalid box')
        if min(cx-w/2,cy-h/2)<-1e-5 or max(cx+w/2,cy+h/2)>1+1e-5:raise ValueError('Box outside image')
        lines.append(f'{int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
    return ''.join(lines)


def main():
    review=ROOT/'annotations/four_class_training_review.json';data=json.loads(review.read_text());dataset=ROOT/'data/v4_four_class_clean'
    backup=ROOT/'artifacts/four-class-study/pre-review-labels';backup.mkdir(parents=True,exist_ok=True)
    edits=[]
    for stem,entry in data['repairs'].items():
        if not re.fullmatch(r'v4_[0-9a-f]{24}',stem):raise ValueError('Unsafe source stem')
        image=dataset/'train/images'/(stem+'.jpg');label=dataset/'train/labels'/(stem+'.txt')
        assert hashlib.sha256(image.read_bytes()).hexdigest()==entry['sha256'],'Reviewed pixels changed'
        current=label.read_text();old=label_text(entry['original_labels']);new=label_text(entry['replacement_labels'])
        if current not in [old,new]:raise ValueError('Training label changed since review: '+stem)
        archived=backup/label.name
        if archived.exists() and archived.read_text()!=old:raise ValueError('Conflicting original label archive')
        edits.append((label,archived,old,new))
    # All seven entries pass preflight before any dataset mutation.
    for label,archived,old,new in edits:
        if not archived.exists():archived.write_text(old)
        if label.read_text()!=new:
            with tempfile.NamedTemporaryFile('w',dir=label.parent,suffix='.review-tmp',delete=False) as stream:
                stream.write(new);temporary=Path(stream.name)
            temporary.replace(label)
    report=validate_dataset(dataset/'data.yaml')
    revised=dict(reviewed_train_images=60,patched_train_images=len(edits),added_boxes=14,added_people_boxes=13,added_sign_boxes=1,
                 review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),human_review_pending=True,
                 control_records_frozen_before_edit=True,validation_test_changed=False,images_changed=False,
                 label_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p,_,_,_ in edits})
    p=dataset/'CLEANING_REPORT.json';clean=json.loads(p.read_text());original=ROOT/'artifacts/four-class-study/cleaning-before-label-review.json'
    if not original.exists():original.write_bytes(p.read_bytes())
    clean['boxes']['train']=dict(zip(report['classes'],report['boxes']['train']));clean['backgrounds']['train']=report['backgrounds']['train'];clean['training_annotation_revision']=revised
    p.write_text(json.dumps(clean,indent=2))
    p=ROOT/'visionaid_config.json';config=json.loads(p.read_text());config['dataset']['training_annotation_revision']=revised
    config['dataset']['version']='v4-four-class-clean-reviewed-2026-11-18';p.write_text(json.dumps(config,indent=2)+'\n')
    (ROOT/'artifacts/four-class-study/applied-label-review.json').write_text(json.dumps(dict(revision=revised,validation=report),indent=2))
    print(json.dumps(revised,indent=2),flush=True)


if __name__=='__main__':main()
