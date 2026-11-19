import os

from PIL import Image

from config import CLASS_NAMES
from evaluation import load_ground_truth
from path_utils import native_path, read_bgr


def test_long_dataset_paths_decode_and_load_labels(tmp_path):
    root = tmp_path / ('video_scene_' + 'x' * 140)
    native_path(root / 'images').mkdir(parents=True)
    native_path(root / 'labels').mkdir()
    Image.new('RGB', (120, 80), (200, 40, 10)).save(native_path(root / 'images/frame.jpg'))
    native_path(root / 'labels/frame.txt').write_text('2 .5 .5 .2 .4\n')
    if os.name == 'nt':
        assert len(str(root / 'images/frame.jpg')) >= 260
    frame = read_bgr(root / 'images/frame.jpg')
    assert frame.shape == (80, 120, 3)
    gt, paths = load_ground_truth(root, CLASS_NAMES)
    assert len(paths) == 1
    assert gt['annotations'][0]['category_id'] == 3
    assert gt['annotations'][0]['bbox'] == [48, 24, 24, 32]
