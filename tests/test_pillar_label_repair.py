import hashlib

import pytest

from pillar_label_repair import repair_label


def decision(original):
    return dict(reviewed_full_image=True,reviewer='AI visual review; team review pending',
        original_label_sha256=hashlib.sha256(original).hexdigest())


def test_reviewed_edit_preserves_other_classes_and_removes_only_the_exact_bad_box():
    original=b'1 0.2 0.2 0.1 0.1\n2 0.5 0.5 0.4 0.4\n'
    review=decision(original)
    review.update(remove_lines=['2 0.5 0.5 0.4 0.4'],
                  add_boxes=[{'class_id':2,'xyxy':[20,10,40,90]}])
    edited,added=repair_label(original,review,100,100)
    assert edited.startswith(b'1 0.2 0.2 0.1 0.1\n')
    assert added==['2 0.300000 0.500000 0.200000 0.800000']
    assert original==b'1 0.2 0.2 0.1 0.1\n2 0.5 0.5 0.4 0.4\n'


def test_unreviewed_proposals_stale_labels_and_outside_boxes_are_rejected():
    original=b''
    review=decision(original)
    with pytest.raises(ValueError,match='changed'):
        repair_label(b'changed',review,100,100)
    review['reviewed_full_image']=False
    with pytest.raises(ValueError,match='adjudication'):
        repair_label(original,review,100,100)


def test_only_line_ending_normalization_is_allowed_when_raw_hash_differs():
    original=b'2 0.5 0.5 0.2 0.2\n'
    review=decision(original)
    review['canonical_label_sha256']=hashlib.sha256(original).hexdigest()
    edited,_=repair_label(original.replace(b'\n',b'\r\n'),review,100,100)
    assert edited==original
    with pytest.raises(ValueError,match='changed'):
        repair_label(original.replace(b'0.2',b'0.3'),review,100,100)
    review['reviewed_full_image']=True
    review['add_boxes']=[dict(class_id=2,xyxy=[-1,0,100,100])]
    with pytest.raises(ValueError,match='outside'):
        repair_label(original,review,100,100)
