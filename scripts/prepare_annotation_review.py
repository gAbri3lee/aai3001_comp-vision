"""Create a training-only annotation review manifest and scene contact sheets.

Every review starts pending. This tool neither corrects labels nor claims a
completed human annotation audit. Originals and validation/test remain intact.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random

NAMES = ['Escalator', 'People', 'Pillar', 'Stair', 'Wet Floor Sign']


def native(path):
    path = path.resolve()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def main():
    from PIL import Image, ImageDraw, ImageOps
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = list(csv.DictReader((root / 'split_manifest.csv').read_text().splitlines()))
    scenes = {r['stem']: r['scene'] for r in manifest if r['split'] == 'train'}
    records, annotations, images, pillar_groups = [], {}, {}, defaultdict(list)
    for index, image in enumerate(sorted((root / 'train/images').iterdir())):
        if image.stem not in scenes:
            raise ValueError('Training image not in manifest')
        label = native(root / 'train/labels' / (image.stem + '.txt'))
        data = label.read_bytes() if label.exists() else b''
        boxes = [list(map(float, row.split())) for row in data.decode().splitlines() if row.strip()]
        counts = Counter(int(b[0]) for b in boxes)
        rid = f'T{index:05d}'
        records.append(dict(review_id=rid, image=image.name, scene=scenes[image.stem],
            existing_classes='|'.join(NAMES[c] for c in sorted(counts)),
            pillar_boxes=counts[2], label_sha256=hashlib.sha256(data).hexdigest(),
            status='pending', reviewer='', scope='', missing_objects='', incorrect_boxes='',
            occlusion='', lighting='', mall_relevance='', decision='', notes=''))
        annotations[rid], images[rid] = boxes, image
        if counts[2]:
            pillar_groups[scenes[image.stem]].append(rid)
    if len(records) != len(scenes):
        raise ValueError('Train manifest membership mismatch')
    # Two size extremes per pillar scene, retaining the whole image for missing-label review.
    selected = []
    for scene, members in sorted(pillar_groups.items()):
        ordered = sorted(members, key=lambda rid: max(b[4] for b in annotations[rid] if int(b[0]) == 2))
        selected.extend(dict.fromkeys([ordered[0], ordered[-1]]))
    negatives = [r['review_id'] for r in records if not r['pillar_boxes']]
    random.Random(0).shuffle(negatives)
    selected += negatives[:48]
    with (args.output / 'training_review.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    by_id = {r['review_id']: r for r in records}
    for start in range(0, len(selected), 24):
        page = Image.new('RGB', (1280, 1800), 'white')
        draw = ImageDraw.Draw(page)
        draw.text((12, 8), 'TRAINING ANNOTATION REVIEW - existing boxes, review pending', fill='black')
        for k, rid in enumerate(selected[start:start + 24]):
            with Image.open(native(images[rid])) as source:
                frame = source.convert('RGB')
            fd = ImageDraw.Draw(frame)
            w, h = frame.size
            for cid, x, y, bw, bh in annotations[rid]:
                box = ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)
                color = 'lime' if int(cid) == 2 else 'orange'
                fd.rectangle(box, outline=color, width=3)
                fd.text((box[0]+2, box[1]+2), NAMES[int(cid)], fill=color)
            tile = ImageOps.contain(frame, (316, 240))
            x, y = (k % 4)*320+2, (k // 4)*294+30
            page.paste(tile, (x, y))
            draw.text((x, y+242), f"{rid} scene: {by_id[rid]['scene'][:30]}", fill='black')
            draw.text((x, y+257), f"pillar boxes: {by_id[rid]['pillar_boxes']} / pending", fill='black')
        page.save(args.output / f'train_review_{start//24:02d}.jpg', quality=90)
    report = dict(split='train', images=len(records), pillar_images=sum(bool(r['pillar_boxes']) for r in records),
        pillar_scenes=len(pillar_groups), contact_sheet_images=len(selected), review_status='pending',
        labels_changed=False, validation_or_test_used=False,
        limitation='Recorded scene groups are not independently verified sites; representative sheets do not exhaust all training images.')
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
