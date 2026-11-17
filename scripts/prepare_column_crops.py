"""Reproduce the reviewed TRAIN-only binary research crops, never edit v4."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from path_utils import native_path


def main():
    from PIL import Image
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/v4_scene_split')
    parser.add_argument('--review',type=Path,default=ROOT/'configs/column_specialist_crop_review.json')
    parser.add_argument('--variant',choices=['initial','hard-negative'],default='initial')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.dataset.resolve();out=args.output.resolve()
    if out.exists() or out==source or source in out.parents:raise ValueError('Use a new directory outside source')
    manifest=(source/'split_manifest.csv').read_bytes();review=json.loads(args.review.read_text())
    if hashlib.md5(manifest).hexdigest()!=review['split_manifest_md5']:raise ValueError('Wrong split')
    members={r['stem'] for r in csv.DictReader(manifest.decode().splitlines()) if r['split']=='train'}
    rows=[r for r in review['rows'] if args.variant=='hard-negative' or r['phase']=='initial']
    out.mkdir(parents=True)
    for row in rows:
        name=row['image']
        if Path(name).name!=name or Path(name).stem not in members:raise ValueError('Only existing TRAIN members allowed')
        path=native_path(source/'train/images'/name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['original_sha256']:raise ValueError('Reviewed pixels changed')
        with Image.open(path) as source_image:
            frame=source_image.convert('RGB');left,top,right,bottom=row['crop_xyxy']
            if not (0<=left<right<=frame.width and 0<=top<bottom<=frame.height):raise ValueError('Invalid reviewed crop')
            crop=frame.crop((left,top,right,bottom))
            if list(crop.size)!=row['size']:raise ValueError('Crop dimensions changed')
            crop.save(out/(row['id']+'.jpg'))
    (out/'review.json').write_text(json.dumps(rows,indent=2))
    (out/'frozen_heldout_groups.json').write_text(json.dumps(review['frozen_validation_source_groups']))
    print(json.dumps(dict(crops=len(rows),variant=args.variant,source_split='train',source_unchanged=True)))


if __name__=='__main__':main()
