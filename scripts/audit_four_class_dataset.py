"""Read-only all-class integrity/duplicate audit of the actual notebook v4 split."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from path_utils import native_path


def inspect_row(item):
    import cv2
    import numpy as np
    from PIL import Image
    split, path, scene = item
    row = dict(split=split, file=path.name, scene=scene, path=str(path.relative_to(ROOT)))
    try:
        data = native_path(path).read_bytes()
        with Image.open(native_path(path)) as image:
            image.load()
            rgb = np.asarray(image.convert('RGB'))
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = (small[:, 1:] > small[:, :-1]).ravel()
        dct = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype('float32'))[:8, :8].copy()
        pb = dct > np.median(dct.ravel()[1:]); pb[0, 0] = False
        labels = native_path(path.parent.parent / 'labels' / (path.stem + '.txt'))
        parsed = []
        for number, line in enumerate(labels.read_text().splitlines() if labels.exists() else [], 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f'label line {number}: expected plain YOLO box')
            cid = int(fields[0]); cx, cy, bw, bh = values = list(map(float, fields[1:]))
            if not 0 <= cid < 5 or not np.isfinite(values).all() or not (0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError(f'label line {number}: invalid class/box')
            if min(cx-bw/2, cy-bh/2) < -1e-5 or max(cx+bw/2, cy+bh/2) > 1+1e-5:
                raise ValueError(f'label line {number}: box outside image')
            parsed.append([cid, *values])
        row.update(width=w, height=h, sha256=hashlib.sha256(data).hexdigest(),
                   pixel_sha256=hashlib.sha256(f'{w},{h},RGB'.encode()+rgb.tobytes()).hexdigest(),
                   dhash=np.packbits(bits).tolist(), phash=np.packbits(pb.ravel()).tolist(),
                   labels=parsed, duplicate_label_rows=len(parsed)-len({tuple(x) for x in parsed}),
                   class_counts=dict(Counter(x[0] for x in parsed)))
        thumb = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
        return row, thumb
    except Exception as exc:
        row['error'] = str(exc)
        return row, None


def groups(rows, key):
    d = defaultdict(list)
    for i, row in enumerate(rows):
        if key in row:
            d[row[key]].append(i)
    return [v for v in d.values() if len(v)>1]


def main():
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'data/v4_scene_split')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/four-class-cleanup/audit.json')
    args=p.parse_args();root=args.dataset.resolve()
    with (root/'split_manifest.csv').open(newline='') as f:
        manifest={r['stem']:r for r in csv.DictReader(f)}
    items=[];orphans=[]
    for split in ['train','valid','test']:
        images=sorted((root/split/'images').iterdir())
        stems={x.stem for x in images}
        orphans.extend(str(x.relative_to(root)) for x in (root/split/'labels').glob('*.txt') if x.stem not in stems)
        items.extend((split,x,manifest[x.stem]['scene']) for x in images)
    rows=[];thumbs=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(r,t) in enumerate(pool.map(inspect_row,items),1):
            rows.append(r);thumbs.append(t)
            if i%2000==0:print(f'Integrity and fingerprints {i}/{len(items)}',flush=True)
    valid=[i for i,r in enumerate(rows) if 'error' not in r]
    hashes=np.array([rows[i]['phash'] for i in valid],dtype=np.uint8)
    dhashes=np.array([rows[i]['dhash'] for i in valid],dtype=np.uint8)
    pop=np.array([x.bit_count() for x in range(256)],dtype=np.uint8)
    pairs=[]
    for start in range(0,len(valid),128):
        distance=pop[hashes[start:start+128,None,:]^hashes[None,:,:]].sum(-1)
        aa,bb=np.nonzero(distance<=4)
        for a,b in zip(aa,bb):
            a+=start
            if a>=b:continue
            i,j=valid[a],valid[b]
            if rows[i]['pixel_sha256']==rows[j]['pixel_sha256']:continue
            dd=int(pop[dhashes[a]^dhashes[b]].sum())
            if dd>6:continue
            mae=float(np.abs(thumbs[i].astype('float32')-thumbs[j]).mean()/255)
            pairs.append(dict(a=i,b=j,phash_distance=int(distance[a-start,b]),dhash_distance=dd,
                              thumbnail_rgb_mae=mae,cross_split=rows[i]['split']!=rows[j]['split'],
                              close_aligned_candidate=mae<=.015))
    scenes=defaultdict(set)
    for r in rows:scenes[r['scene']].add(r['split'])
    same_stem=defaultdict(set)
    for r in rows:same_stem[Path(r['file']).stem.split('.rf.')[0]].add(r['split'])
    exact=groups(rows,'sha256');pixels=groups(rows,'pixel_sha256')
    summary=dict(images=len(rows),splits=dict(Counter(r['split'] for r in rows)),
        corrupt_or_invalid=sum('error' in r for r in rows),orphan_labels=len(orphans),
        duplicate_label_rows=sum(r.get('duplicate_label_rows',0) for r in rows),
        byte_duplicate_groups=len(exact),pixel_duplicate_groups=len(pixels),
        byte_cross_split_groups=sum(len({rows[i]['split'] for i in g})>1 for g in exact),
        pixel_cross_split_groups=sum(len({rows[i]['split'] for i in g})>1 for g in pixels),
        perceptual_candidates=len(pairs),perceptual_cross_split_candidates=sum(x['cross_split'] for x in pairs),
        close_aligned_candidates=sum(x['close_aligned_candidate'] for x in pairs),
        scene_cross_split=sum(len(s)>1 for s in scenes.values()),
        source_stem_cross_split=sum(len(s)>1 for s in same_stem.values()))
    report=dict(summary=summary,rows=rows,byte_duplicate_groups=exact,pixel_duplicate_groups=pixels,
                perceptual_pairs=pairs,orphan_labels=orphans,
                methodology='Decode every image, validate every label, SHA256 bytes+RGB pixels, all-class pHash<=4 and dHash<=6. Near pairs are candidates, never certified duplicates from hashes alone.',
                limitation='Cannot certify visual annotation completeness, all same-location scenes, crop/rotation overlap or pretraining independence.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
