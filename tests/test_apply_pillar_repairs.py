from pathlib import Path
import sys

import pytest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),
             str(Path(__file__).resolve().parents[1]/'src')]
from apply_pillar_repairs import copy_validation_split
from path_utils import native_path


def test_validation_copy_keeps_long_windows_filenames_and_exact_bytes(tmp_path):
    root=tmp_path/'source'
    target=tmp_path/'target'
    name='frame_'+'x'*195
    for kind,suffix,data in [('images','.jpg',b'pixel bytes'),('labels','.txt',b'2 .5 .5 .2 .8\r\n')]:
        folder=root/'valid'/kind
        folder.mkdir(parents=True)
        native_path(folder/(name+suffix)).write_bytes(data)
    assert len(str(root/'valid/images'/(name+'.jpg'))) > 260
    assert copy_validation_split(root,target,{name}) == 1
    assert native_path(target/'valid/images'/(name+'.jpg')).read_bytes()==b'pixel bytes'
    assert native_path(target/'valid/labels'/(name+'.txt')).read_bytes()==b'2 .5 .5 .2 .8\r\n'


def test_validation_copy_rejects_missing_manifest_member(tmp_path):
    (tmp_path/'source/valid/images').mkdir(parents=True)
    with pytest.raises(ValueError,match='membership'):
        copy_validation_split(tmp_path/'source',tmp_path/'target',{'missing'})
