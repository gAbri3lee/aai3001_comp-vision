"""Validate notebook-prepared YOLO data without downloading or changing its split."""
import csv
import hashlib
import io
import math
from pathlib import Path

import yaml

from config import CLASS_NAMES
from detection import validate_names

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def validate_dataset(data_yaml, expected_fingerprint=None):
    """Return counts/provenance, rejecting old classes and overlapping splits.

    The notebooks export data.yaml and split_manifest.csv alongside train/valid/
    test. Missing label files are background images, as in their evaluator.
    This verifies manifest membership and scene separation, not visual duplicates.
    """
    data_yaml = Path(data_yaml).expanduser().resolve()
    spec = yaml.safe_load(data_yaml.read_text())
    if not isinstance(spec, dict):
        raise ValueError('Dataset YAML must be a mapping')
    try:
        validate_names(spec.get('names', []), CLASS_NAMES)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Dataset classes must match {list(CLASS_NAMES)!r} in ID order') from exc
    if spec.get('nc', 5) != 5:
        raise ValueError('Dataset classes must match nc=5')
    root = Path(spec.get('path') or '.').expanduser()
    root = (data_yaml.parent / root).resolve()
    splits = {}
    for split in ('train', 'val', 'test'):
        entry = spec.get(split)
        if not isinstance(entry, str) or not entry:
            raise ValueError(f'{split}: expected a notebook-exported images directory')
        directory = (root / entry).resolve()
        if directory.name != 'images' or not directory.is_dir():
            raise ValueError(f'Missing {split} images directory: {directory}; '
                             'set the YAML path to the relocated notebook dataset root')
        splits[split] = directory
    if len(set(splits.values())) != 3:
        raise ValueError('Dataset splits overlap; a test-only handover is for evaluation, not training')

    counts, backgrounds, boxes = {}, {}, {}
    membership, seen_paths = {}, set()
    for split, directory in splits.items():
        entries = list(directory.iterdir())
        if any(p.name != '.DS_Store' and (not p.is_file() or p.name.startswith('.') or
                p.suffix.lower() not in IMAGE_SUFFIXES) for p in entries):
            raise ValueError(f'{split}: expected flat notebook images only; '
                             'nested, hidden or unsupported files can differ from the training loader')
        images = sorted(p for p in entries if p.name != '.DS_Store')
        if not images:
            raise ValueError(f'{split}: no images found')
        labels = directory.parent / 'labels'
        if not labels.is_dir():
            raise ValueError(f'Missing {split} labels directory: {labels}')
        counts[split], backgrounds[split], boxes[split] = len(images), 0, [0] * 5
        stems = set()
        for image in images:
            if image.stem in membership or image.resolve() in seen_paths:
                raise ValueError(f'Dataset image overlap: {image.name}')
            membership[image.stem] = split
            seen_paths.add(image.resolve())
            stems.add(image.stem)
            label = labels / (image.stem + '.txt')
            lines = [line for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
            backgrounds[split] += not lines
            for number, line in enumerate(lines, 1):
                try:
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError('expected class ID and four box values (convert polygons in the notebook)')
                    cid = int(fields[0])
                    cx, cy, width, height = map(float, fields[1:])
                    if not 0 <= cid < 5:
                        raise ValueError('class ID must be between 0 and 4')
                    if (not all(math.isfinite(v) for v in (cx, cy, width, height)) or
                            not 0 < width <= 1 or not 0 < height <= 1 or
                            cx - width / 2 < -1e-5 or cx + width / 2 > 1 + 1e-5 or
                            cy - height / 2 < -1e-5 or cy + height / 2 > 1 + 1e-5):
                        raise ValueError('box must be finite, normalized and within the image')
                except ValueError as exc:
                    raise ValueError(f'Invalid label {label}:{number}: {exc}') from exc
                boxes[split][cid] += 1
        if any(p.stem not in stems for p in labels.glob('*.txt')):
            raise ValueError(f'{split}: orphan label without a matching image')
        if not all(boxes[split]):
            raise ValueError(f'{split}: labels must include all five classes')

    manifest = root / 'split_manifest.csv'
    if not manifest.is_file():
        raise ValueError('Missing split_manifest.csv; export the prepared scene split from a training notebook')
    text = manifest.read_text()
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != ['stem', 'scene', 'split']:
        raise ValueError('Invalid split manifest columns; expected stem,scene,split')
    manifest_membership, scenes = {}, {}
    for row in reader:
        stem, scene, split = row['stem'], row['scene'], row['split']
        split = 'val' if split == 'valid' else split
        if not stem or not scene or split not in splits or stem in manifest_membership:
            raise ValueError('Invalid or duplicate split manifest entry')
        if scene in scenes and scenes[scene] != split:
            raise ValueError(f'Split manifest scene overlaps splits: {scene}')
        scenes[scene], manifest_membership[stem] = split, split
    if manifest_membership != membership:
        raise ValueError('Split manifest does not match the actual images/splits')
    fingerprint = hashlib.md5(text.encode()).hexdigest()
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError(f'Split fingerprint does not match finalized notebook split: {fingerprint}')
    resolved_yaml = dict(path=str(root), nc=5, names=list(CLASS_NAMES),
                         **{split: str(directory) for split, directory in splits.items()})
    return dict(data=str(data_yaml), classes=list(CLASS_NAMES), images=counts,
                boxes=boxes, backgrounds=backgrounds, split_fingerprint=fingerprint,
                resolved_yaml=resolved_yaml)
