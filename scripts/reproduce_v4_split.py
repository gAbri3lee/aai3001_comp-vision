"""Reproduce the repository notebook's v4 scene split without downloading or training.

Only literal split settings and the inspected split-calculation cell are used.
The notebook's package installation, credential handling, deletion and training
cells are never executed. Originals are preserved; output must not exist.
"""
import ast
import collections
from concurrent.futures import ThreadPoolExecutor
import glob
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]


def native(p):
    text = str(Path(p).absolute())
    return Path(text if text.startswith('\\\\?\\') or os.name != 'nt' else '\\\\?\\' + text)


def main():
    raw = ROOT / 'data/raw_v4_export'
    output = ROOT / 'data/v4_scene_split'
    if output.exists():
        raise ValueError('Refusing to overwrite an existing split')
    notebook = json.loads((ROOT / 'yolov8n2.ipynb').read_text(encoding='utf-8'))
    settings = ''.join(notebook['cells'][1]['source'])
    wanted = {'SEED', 'GROUP_OVERRIDES', 'EASY_SOURCES', 'SPLIT_FRAC', 'PRIORITY_CLASSES',
              'PRIORITY_FRAC', 'DROP_UNLABELLED_HINTS', 'HINTS', 'BG_KEEP', 'DUP_HAMMING', 'TRIES'}
    namespace = dict(os=os, re=re, glob=glob, json=json, random=random, hashlib=hashlib,
                     collections=collections, np=np, pd=pd, Image=Image, yaml=yaml,
                     DATA_ROOT=str(native(raw)), SPLITS=('train', 'valid', 'test'),
                     IMG_EXT=('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    for node in ast.parse(settings).body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in wanted:
            namespace[node.targets[0].id] = ast.literal_eval(node.value)
        elif len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            names = [n.id for n in node.targets[0].elts]
            if all(n in wanted for n in names):
                namespace.update(zip(names, ast.literal_eval(node.value)))
    if not wanted.issubset(namespace):
        raise ValueError('Notebook split settings changed; inspect before continuing')
    random.seed(namespace['SEED'])
    np.random.seed(namespace['SEED'])
    calculation = ''.join(notebook['cells'][4]['source']).split('if os.path.exists(DATA_DIR):')[0]
    print('Calculating original scene split from raw v4...', flush=True)
    exec(compile(calculation, 'inspected_v4_split_calculation', 'exec'), namespace)
    rows = namespace['NEW']
    manifest = rows[['stem', 'scene', 'split']].sort_values('stem')
    text = manifest.to_csv(index=False, lineterminator='\n')
    fingerprint = hashlib.md5(text.encode()).hexdigest()
    expected = json.loads((ROOT / 'visionaid_config.json').read_text())['dataset']['split_fingerprint']
    supplied = {p.stem for p in (ROOT / 'artifacts/supplied_weights/test_set/test/images').glob('*.jpg')}
    actual = set(rows[rows.split == 'test'].stem)
    summary = dict(images={s: int((rows.split == s).sum()) for s in namespace['SPLITS']},
                   split_fingerprint=fingerprint, expected_fingerprint=expected,
                   supplied_test_membership_matches=actual == supplied,
                   extra_test=len(actual-supplied), missing_test=len(supplied-actual))
    (ROOT / 'artifacts/v4_split_reproduction.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if fingerprint != expected or actual != supplied:
        raise ValueError('Split does not reproduce the trained model protocol; no training output written')
    output.mkdir(parents=True)
    for s in namespace['SPLITS']:
        for kind in ('images', 'labels'):
            (output / s / kind).mkdir(parents=True)
    def copy(row):
        shutil.copy2(row.image, native(output / row.split / 'images' / Path(row.image).name))
        if row.labels:
            lines = []
            for c, (x1, y1, x2, y2) in row.labels:
                x1, y1, x2, y2 = max(0., x1), max(0., y1), min(1., x2), min(1., y2)
                lines.append(f'{c} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}\n')
            native(output / row.split / 'labels' / (row.stem + '.txt')).write_text(''.join(lines))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(copy, rows.itertuples()), 1):
            if i % 2000 == 0:
                print(f'Copied {i}/{len(rows)} verified split members', flush=True)
    (output / 'split_manifest.csv').write_text(text, encoding='utf-8', newline='')
    spec = dict(path=str(output), train='train/images', val='valid/images', test='test/images',
                nc=5, names=namespace['NAMES'])
    (output / 'data.yaml').write_text(yaml.safe_dump(spec, sort_keys=False), encoding='utf-8')
    print('Original v4 split reproduced:', output, flush=True)


if __name__ == '__main__':
    main()
