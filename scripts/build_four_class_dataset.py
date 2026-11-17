"""Create an immutable, traceable four-class derivative; never edit v4 originals.

Similar-frame candidates are conservatively quarantined, not all called proven
duplicates. Split membership is preserved; images containing removed-class
annotations remain useful background/other-class examples.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from path_utils import native_path
NAMES=['Escalator','People','Stair','Wet Floor Sign']
MAPPING={0:0,1:1,3:2,4:3}


def select_rows(audit):
    """Greedy independent set: no retained pair in the specified candidate graph.

    Prefer informative images (retained label count), then resolution, then
    stable names. No model predictions, confidence or evaluation scores used.
    Originals and every exclusion reason remain available for reversal.
    """
    rows=audit['rows'];neighbors=defaultdict(set)
    for key in ['byte_duplicate_groups','pixel_duplicate_groups']:
        for group in audit[key]:
            for i in group:
                neighbors[i].update(set(group)-{i})
    for pair in audit['perceptual_pairs']:
        i,j=pair['a'],pair['b'];neighbors[i].add(j);neighbors[j].add(i)
    priority={'train':0,'valid':1,'test':2}
    order=sorted(range(len(rows)),key=lambda i:(priority[rows[i]['split']],
        -sum(t[0] in MAPPING for t in rows[i].get('labels',[])),
        -rows[i].get('width',0)*rows[i].get('height',0),rows[i]['file']))
    kept=set();excluded=[]
    for i in order:
        if 'error' in rows[i]:
            excluded.append(dict(index=i,reason='invalid_image_or_label',detail=rows[i]['error']));continue
        matches=neighbors[i]&kept
        if matches:
            j=min(matches)
            reason='exact_duplicate' if rows[i]['pixel_sha256']==rows[j]['pixel_sha256'] else 'similar_frame_candidate_quarantined'
            excluded.append(dict(index=i,reason=reason,retained_index=j));continue
        kept.add(i)
    assert all(not ({p['a'],p['b']}<=kept) for p in audit['perceptual_pairs'])
    return kept,excluded


def main():
    import yaml
    audit_path=ROOT/'artifacts/four-class-cleanup/audit.json';audit=json.loads(audit_path.read_text())
    output=ROOT/'data/v4_four_class_clean'
    if output.exists():raise ValueError('Refusing to overwrite existing cleaned dataset')
    kept,excluded=select_rows(audit);rows=audit['rows']
    if audit['summary']['scene_cross_split']:raise ValueError('Resolve scene leakage before creating derivative')
    for split in ['train','valid','test']:
        for kind in ['images','labels']:(output/split/kind).mkdir(parents=True,exist_ok=True)
    manifest=[];provenance=[];counts=Counter();boxes=defaultdict(Counter);backgrounds=Counter();removed=Counter();seen=set()
    for i in sorted(kept):
        row=rows[i];split=row['split'];source=ROOT/row['path'];stem='v4_'+row['sha256'][:24]
        if stem in seen:raise ValueError('Unexpected short-hash collision')
        seen.add(stem);name=stem+source.suffix.lower();target=output/split/'images'/name
        assert hashlib.sha256(native_path(source).read_bytes()).hexdigest()==row['sha256'],'Source changed after audit'
        shutil.copy2(native_path(source),target)
        selected=[]
        for cid,cx,cy,bw,bh in row['labels']:
            if cid==2:removed[split]+=1;continue
            selected.append([MAPPING[cid],cx,cy,bw,bh]);boxes[split][NAMES[MAPPING[cid]]]+=1
        # Explicit empty labels avoid ambiguity between missing files and negatives.
        (output/split/'labels'/(stem+'.txt')).write_text(''.join(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for c,x,y,w,h in selected))
        counts[split]+=1;backgrounds[split]+=not selected
        manifest.append(dict(stem=stem,scene=row['scene'],split=split))
        provenance.append(dict(stem=stem,original_file=row['file'],original_path=row['path'],split=split,sha256=row['sha256'],
                               pillar_boxes_removed=sum(t[0]==2 for t in row['labels'])))
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=['stem','scene','split'],lineterminator='\n');w.writeheader();w.writerows(sorted(manifest,key=lambda x:x['stem']))
    text=stream.getvalue();(output/'split_manifest.csv').write_text(text,newline='')
    # Omit path: Ultralytics then resolves relative split paths against the YAML
    # directory. An explicit '.' would resolve against the launch directory.
    (output/'data.yaml').write_text(yaml.safe_dump(dict(train='train/images',val='valid/images',test='test/images',nc=4,names=NAMES),sort_keys=False))
    for name in ['README.dataset.txt','README.roboflow.txt']:
        source=ROOT/'data/raw_v4_export'/name
        if source.is_file():shutil.copy2(source,output/('SOURCE_'+name))
    (output/'README.clean.txt').write_text(
        'Four-class derivative, 17 November 2026. Active names/IDs are in data.yaml. '
        'Pillar annotations removed; background pillars can remain. Original source readmes '
        'retain supplier attribution and historical counts, not current class definitions. '
        'provenance.csv traces originals; CLEANING_REPORT.json records exclusions. '
        'No retraining or exhaustive independent visual reannotation is claimed.\n')
    with (output/'provenance.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(provenance[0]));writer.writeheader();writer.writerows(provenance)
    report=dict(source='data/v4_scene_split',output='data/v4_four_class_clean',classes=NAMES,old_to_new=MAPPING,
        images=dict(counts),boxes={s:dict(v) for s,v in boxes.items()},backgrounds=dict(backgrounds),
        pillar_boxes_removed_in_retained_images=dict(removed),pillar_annotations_remaining=0,
        quarantined_images=len(excluded),quarantine_reasons=dict(Counter(x['reason'] for x in excluded)),
        excluded=[dict(**x,source=rows[x['index']]['path']) for x in excluded],
        fingerprint=hashlib.md5(text.encode()).hexdigest(),source_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        splits_moved=0,byte_and_pixel_duplicates_remaining=0,retained_audited_perceptual_pairs=0,
        policy='Remove pillar labels, retain useful photos and negatives, preserve original scene split. Candidate-similar images quarantined to originals; not all candidates are certified duplicates. Short content-derived names avoid Windows path failures.',
        limitation='No exhaustive human reannotation of every image. Unlabelled targets, unrelated viewpoints/crops and background pillars may remain. This is a new dataset version; old model metrics are historical.')
    (output/'CLEANING_REPORT.json').write_text(json.dumps(report,indent=2))
    (ROOT/'artifacts/four-class-cleanup/dataset-build.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='excluded'},indent=2),flush=True)


if __name__=='__main__':main()
