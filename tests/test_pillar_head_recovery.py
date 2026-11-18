"""Candidate mining must not turn ambiguous proposals into hard negatives."""
import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location('pillar_head_recovery',
    Path(__file__).resolve().parents[1]/'scripts/pillar_head_recovery.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_mining_uses_overlap_truth_and_ignores_ambiguous_boxes():
    boxes = torch.tensor([[0.,0.,10.,10.],[0.,0.,5.,5.],[20.,20.,30.,30.]])
    truths = torch.tensor([[0.,0.,10.,10.]])
    chosen,target,coverage = recovery.select_training_anchors(boxes,truths,
        torch.tensor([-8.,8.,4.]),[1,1,1],0)
    assert chosen.tolist() == [0,2]  # 0.25 IoU is ignored despite high score.
    assert target.tolist() == [1.,0.]
    assert coverage == {'ground_truth_pillars':1,'covered_truths':1}


def test_backgrounds_produce_only_negatives_and_selection_is_repeatable():
    boxes = torch.arange(400.).reshape(100,4)
    empty = torch.empty((0,4))
    first = recovery.select_training_anchors(boxes,empty,torch.arange(100.),[60,30,10],42)
    second = recovery.select_training_anchors(boxes,empty,torch.arange(100.),[60,30,10],42)
    assert torch.equal(first[0],second[0])
    assert not first[1].any()
    assert first[2]['ground_truth_pillars'] == 0
    assert len(first[0]) <= 96
