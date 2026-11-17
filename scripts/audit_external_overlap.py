"""Screen online images against every v4 image; near hashes need visual review."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from path_utils import native_path


def dhash(image):
    values = list(image.convert('L').resize((9, 8)).getdata())
    return sum((values[y*9+x] > values[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))


def fingerprint(item):
    from PIL import Image
    split, path = item
    data = native_path(path).read_bytes()
    with Image.open(native_path(path)) as image:
        value = dhash(image)
    return dict(split=split, image=path.name, sha256=hashlib.sha256(data).hexdigest(), dhash=value)


def main():
    from PIL import Image, ImageOps
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/v4_scene_split'))
    parser.add_argument('--sources', type=Path, default=Path('data/online_sg_malls'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/online-overlap.json'))
    args = parser.parse_args()
    paths = [(split,p) for split in ['train','valid','test']
             for p in (args.dataset / split / 'images').iterdir()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        original = list(pool.map(fingerprint, paths))
    rows = []
    for r in json.loads((args.sources / 'sources.json').read_text()):
        if Path(r.get('file','')).suffix.lower() not in ('.jpg','.jpeg','.png'):
            continue
        with Image.open(args.sources / r['file']) as im:
            hashes = [dhash(im), dhash(ImageOps.pad(im.convert('RGB'), (640,640), color='black'))]
        nearest = min(original, key=lambda p: min((p['dhash']^h).bit_count() for h in hashes))
        distance = min((nearest['dhash']^h).bit_count() for h in hashes)
        rows.append(dict(id=r['id'], exact_matches=[{k:p[k] for k in ('split','image')} for p in original
            if p['sha256']==r['sha256']], nearest={k:nearest[k] for k in ('split','image')},
            nearest_dhash_distance=distance, visual_review_required=distance<=6))
    report = dict(v4_images=len(original), online_images=len(rows), rows=rows,
        limitation='Heuristic duplicate screen only. Different crops/black bars can evade it; low-detail images can collide. Does not certify venue/pretraining independence or physical field validity.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
