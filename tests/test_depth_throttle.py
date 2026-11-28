"""Live depth runs on every Nth frame and reuses its map in between, never across sources or sizes."""
import numpy as np
import pytest

from depth import ThrottledDepth


class Estimator:
    def __init__(self):
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        return np.full(frame.shape[:2], self.calls, dtype=float)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


def test_every_third_call_runs_the_model():
    estimator, clock = Estimator(), Clock()
    depth = ThrottledDepth(estimator, every=3, clock=clock)
    maps = []
    for _ in range(7):
        maps.append(depth.predict(FRAME)[0, 0])
        clock.now += .05
    assert maps == [1, 1, 1, 2, 2, 2, 3]
    assert (depth.computed, depth.reused) == (3, 4)


def test_every_one_is_the_original_behaviour():
    estimator = Estimator()
    depth = ThrottledDepth(estimator, every=1, clock=Clock())
    assert [depth.predict(FRAME)[0, 0] for _ in range(3)] == [1, 2, 3]


def test_old_map_is_recomputed():
    estimator, clock = Estimator(), Clock()
    depth = ThrottledDepth(estimator, every=5, max_age=1.0, clock=clock)
    depth.predict(FRAME)
    clock.now = 1.5            # no detections for a while, then a hazard appears
    assert depth.predict(FRAME)[0, 0] == 2


def test_new_frame_size_or_reset_recomputes():
    estimator = Estimator()
    depth = ThrottledDepth(estimator, every=5, clock=Clock())
    depth.predict(FRAME)
    other = depth.predict(np.zeros((30, 40, 3), dtype=np.uint8))
    assert other.shape == (30, 40) and other[0, 0] == 2
    depth.reset()
    assert depth.predict(np.zeros((30, 40, 3), dtype=np.uint8))[0, 0] == 3


@pytest.mark.parametrize('every, max_age', [(0, 1.), (1.5, 1.), (2, 0)])
def test_invalid_settings_are_rejected(every, max_age):
    with pytest.raises(ValueError):
        ThrottledDepth(Estimator(), every=every, max_age=max_age)
