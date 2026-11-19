"""Build an isolated training variant from confirmed scene-quarantine decisions.

Original images/labels and validation/test membership are never changed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from config import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--decisions', type=Path, default=ROOT / 'configs/v4_training_quarantine.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    policy = json.loads(args.decisions.read_text())
    manifest = (root / 'split_manifest.csv').read_bytes()
    if hashlib.md5(manifest).hexdigest() != policy['source_split_fingerprint']:
        raise ValueError('Source scene split differs from the reviewed v4 split')
    rows = list(csv.DictReader(manifest.decode().splitlines()))
    decisions = {d['scene']: d for d in policy['scenes']}
    for scene, d in decisions.items():
        affected = [r for r in rows if r['scene'] == scene]
        if not affected or any(r['split'] != 'train' for r in affected):
            raise ValueError('Quarantine must affect existing training scenes only')
        evidence = root / 'train/labels' / (Path(d['evidence_image']).stem + '.txt')
        if d['evidence_label'] not in evidence.read_text().splitlines():
            raise ValueError('Reviewed evidence changed; repeat annotation review')
    train = {p.stem: p for p in (root / 'train/images').iterdir()}
    rows = [r for r in rows if r['split'] == 'train']
    if set(train) != {r['stem'] for r in rows}:
        raise ValueError('Training directory and manifest differ')
    kept = [train[r['stem']] for r in rows if r['scene'] not in decisions]
    removed = [r for r in rows if r['scene'] in decisions]
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    all_list, clean_list = output / 'train_original.txt', output / 'train_curated.txt'
    all_list.write_text(''.join(str(p) + '\n' for p in train.values()))
    clean_list.write_text(''.join(str(p) + '\n' for p in kept))
    for name, listing in [('control', all_list), ('curated', clean_list)]:
        (output / (name + '.yaml')).write_text(yaml.safe_dump(dict(path=str(root),
            train=str(listing), val='valid/images', nc=5, names=list(CLASS_NAMES))))
    changes = [dict(stem=r['stem'], scene=r['scene'], action='quarantine_training_scene',
                    reason=decisions[r['scene']]['reason']) for r in removed]
    with (output / 'annotation_changes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['stem', 'scene', 'action', 'reason'])
        writer.writeheader()
        writer.writerows(changes)
    report = dict(version=policy['version'], original_train=len(train), curated_train=len(kept),
        quarantined_images=len(removed), quarantined_scenes=list(decisions),
        source_manifest_md5=hashlib.md5(manifest).hexdigest(),
        curated_train_list_sha256=hashlib.sha256(clean_list.read_bytes()).hexdigest(),
        decisions_sha256=hashlib.sha256(args.decisions.read_bytes()).hexdigest(),
        validation_changed=False, test_changed=False, original_labels_changed=False,
        model_trained=False, annotation_audit_complete=False,
        note='A training candidate, not an improved model or a complete annotation repair.')
    (output / 'curation_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
