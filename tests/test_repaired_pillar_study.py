from collections import Counter
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from repaired_pillar_study import schedule_draws


def test_matched_schedule_excludes_unreviewed_frames_and_caps_scene_exposure():
    positives = {f'p{i}.jpg' for i in range(100)} | {'single.jpg'}
    negatives = {f'n{i}.jpg' for i in range(1000)}
    scenes = {Path(n).stem:('large' if n.startswith('p') else n) for n in positives|negatives}
    images = positives|negatives|{'excluded.jpg'}
    draws = schedule_draws(images, scenes, positives, {'excluded.jpg'}, total=500)
    assert len(draws) == 500
    assert 'excluded.jpg' not in draws
    assert sum(n.startswith('p') for n in draws) == 32
    assert draws.count('single.jpg') == 4
    assert max(Counter(draws).values()) <= 4
    assert len({n for n in draws if n in negatives}) == 464
    assert draws == schedule_draws(images,scenes,positives,{'excluded.jpg'},total=500)


def test_insufficient_negatives_rejected_instead_of_unbounded_resampling():
    with pytest.raises(ValueError, match='unique negatives'):
        schedule_draws(['p.jpg','n.jpg'],{'p':'a','n':'b'},{'p.jpg'},set(),total=100)
