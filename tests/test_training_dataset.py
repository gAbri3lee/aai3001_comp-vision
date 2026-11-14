"""Validate prepared notebook datasets without training or downloading data."""
import hashlib
import json

import pytest
import yaml

from config import CLASS_NAMES, load_config
from training_dataset import validate_dataset


@pytest.fixture
def prepared_dataset(tmp_path):
    manifest = 'stem,scene,split\n'
    for split in ('train', 'valid', 'test'):
        (tmp_path / split / 'images').mkdir(parents=True)
        (tmp_path / split / 'labels').mkdir()
        # Five labelled images plus a background lacking a label file.
        for cid in range(6):
            stem = f'{split}_{cid}'
            (tmp_path / split / 'images' / f'{stem}.jpg').write_bytes(b'fixture')
            if cid < 5:
                (tmp_path / split / 'labels' / f'{stem}.txt').write_text(
                    f'{cid} 0.5 0.5 0.2 0.2\n')
            manifest += f'{stem},{split}_scene,{split}\n'
    (tmp_path / 'split_manifest.csv').write_text(manifest)
    path = tmp_path / 'data.yaml'
    path.write_text(yaml.safe_dump(dict(path='.', train='train/images',
        val='valid/images', test='test/images', nc=5, names=list(CLASS_NAMES))))
    config = load_config()
    config['dataset']['split_fingerprint'] = hashlib.md5(manifest.encode()).hexdigest()
    (tmp_path / 'config.json').write_text(json.dumps(config))
    return path


@pytest.mark.parametrize('extra', ['nested/test_0.jpg', 'untracked.tiff'])
def test_rejects_files_loader_would_use_outside_manifest(prepared_dataset, extra):
    path = prepared_dataset.parent / 'train/images' / extra
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b'untracked image')
    with pytest.raises(ValueError, match='flat notebook images'):
        validate_dataset(prepared_dataset)


def test_training_descriptor_uses_absolute_validated_paths(prepared_dataset):
    descriptor = validate_dataset(prepared_dataset)['resolved_yaml']
    assert descriptor['path'] == str(prepared_dataset.parent)
    assert descriptor['train'] == str(prepared_dataset.parent / 'train/images')
    assert descriptor['val'] == str(prepared_dataset.parent / 'valid/images')
    assert descriptor['test'] == str(prepared_dataset.parent / 'test/images')


def test_prepared_split_preserves_classes_backgrounds_and_fingerprint(prepared_dataset):
    report = validate_dataset(prepared_dataset)
    assert report['classes'] == ['Escalator', 'People', 'Pillar', 'Stair', 'Wet Floor Sign']
    assert report['images'] == {'train': 6, 'val': 6, 'test': 6}
    assert report['boxes']['train'] == [1, 1, 1, 1, 1]
    assert report['backgrounds'] == {'train': 1, 'val': 1, 'test': 1}
    assert report['split_fingerprint'] == json.loads(
        (prepared_dataset.parent / 'config.json').read_text())['dataset']['split_fingerprint']


@pytest.mark.parametrize('names', [
    ['Person', 'Car', 'Chair'],
    ['People', 'Escalator', 'Pillar', 'Stair', 'Wet Floor Sign'],
])
def test_old_or_reordered_classes_rejected_before_training(prepared_dataset, names):
    spec = yaml.safe_load(prepared_dataset.read_text())
    spec['names'], spec['nc'] = names, len(names)
    prepared_dataset.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='Dataset classes must match'):
        validate_dataset(prepared_dataset)


def test_test_only_bundle_is_rejected(prepared_dataset):
    spec = yaml.safe_load(prepared_dataset.read_text())
    spec.update(train='test/images', val='test/images')
    prepared_dataset.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='overlap'):
        validate_dataset(prepared_dataset)


@pytest.mark.parametrize('label', ['5 .5 .5 .2 .2', '2 nan .5 .2 .2',
    '2 .5 .5 -.2 .2', '2 .5 .5 1.5 .2', '2 .5 .5 .2 .2 .3 .3'])
def test_invalid_labels_rejected(prepared_dataset, label):
    (prepared_dataset.parent / 'train/labels/train_2.txt').write_text(label)
    with pytest.raises(ValueError, match='label'):
        validate_dataset(prepared_dataset)


def test_manifest_scene_leakage_rejected(prepared_dataset):
    manifest = prepared_dataset.parent / 'split_manifest.csv'
    manifest.write_text(manifest.read_text().replace('valid_scene', 'train_scene'))
    with pytest.raises(ValueError, match='scene'):
        validate_dataset(prepared_dataset)


def test_manifest_must_cover_actual_images(prepared_dataset):
    (prepared_dataset.parent / 'test/images/test_5.jpg').unlink()
    with pytest.raises(ValueError, match='manifest'):
        validate_dataset(prepared_dataset)


def test_changed_fingerprint_blocks_training(prepared_dataset):
    config_path = prepared_dataset.parent / 'config.json'
    config = json.loads(config_path.read_text())
    config['dataset']['split_fingerprint'] = 'wrong'
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Split fingerprint does not match'):
        validate_dataset(prepared_dataset, config['dataset']['split_fingerprint'])
