import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_whole_scene_protocol_has_disjoint_venues_and_valid_unchanged_pixels():
    from PIL import Image
    protocol=json.loads((ROOT/'annotations/mall_whole_scene_columns.json').read_text())
    if any(not (ROOT/row['path']).exists() for row in protocol['images']):
        import pytest
        pytest.skip('Licensed research images must be downloaded for the pixel-integrity check')
    groups={};ids=set()
    for row in protocol['images']:
        assert row['id'] not in ids
        ids.add(row['id']);groups.setdefault(row['venue'],set()).add(row['split'])
        path=ROOT/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with Image.open(path) as image:w,h=image.size
        for x1,y1,x2,y2 in row['boxes']:assert 0<=x1<x2<=w and 0<=y1<y2<=h
    assert all(len(s)==1 for s in groups.values())
    assert protocol['predeclared']['checkpoint'].startswith('last;')


def test_controlled_dataset_has_equal_draws_and_identical_validation():
    root=ROOT/'artifacts/mall-whole-scene-study'
    if not root.exists():
        import pytest
        pytest.skip('Generated study datasets unavailable')
    control=root/'standard/dataset';augmented=root/'degradation_augmented/dataset'
    assert {p.name for p in (control/'train/images').glob('*.jpg')}=={p.name for p in (augmented/'train/images').glob('*.jpg')}
    for split in ['train','val']:
        for p in (control/split/'labels').glob('*.txt'):assert p.read_bytes()==(augmented/split/'labels'/p.name).read_bytes()
    for p in (control/'val/images').glob('*.jpg'):assert p.read_bytes()==(augmented/'val/images'/p.name).read_bytes()
    assert not any(p.name.startswith('F') for p in control.rglob('*.jpg'))
