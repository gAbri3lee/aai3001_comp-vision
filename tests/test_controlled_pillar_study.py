import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from controlled_pillar_study import balanced_draws, acceptance


def test_sampling_has_matched_budget_and_limits_single_image_repetition():
    groups = {'small': ['a'], 'large': ['b', 'c', 'd']}
    negatives = [f'n{i}' for i in range(18)]
    draws = balanced_draws(groups, negatives, total=20, seed=0)
    assert len(draws) == 20
    assert sum(p in {'a', 'b', 'c', 'd'} for p in draws) == 5
    assert max(Counter(p for p in draws if p in {'a', 'b', 'c', 'd'}).values()) <= 4
    assert draws == balanced_draws(groups, negatives, total=20, seed=0)


def test_sampling_fails_instead_of_exceeding_pillar_repeat_cap():
    with pytest.raises(ValueError, match='repeat cap'):
        balanced_draws({'one': ['a']}, list(range(30)), total=24, seed=0, fraction=.5)


def test_gate_rejects_recall_gains_bought_with_false_positives_or_other_class_loss():
    from copy import deepcopy
    names = ['Escalator','People','Pillar','Stair','Wet Floor Sign']
    before = {'map50':.72,'per_class':{n:{'precision':.8,'recall':.6} for n in names}}
    before['per_class']['Pillar']['recall'] = .29
    after = deepcopy(before)
    after['per_class']['Pillar']['recall'] = .42
    assert acceptance(before,after) == (True,[])
    after['per_class']['Pillar']['precision'] = .55
    assert acceptance(before,after) == (False,[])
    after['per_class']['Pillar']['precision'] = .8
    after['per_class']['Stair']['recall'] = .55
    assert acceptance(before,after) == (False,['Stair'])
