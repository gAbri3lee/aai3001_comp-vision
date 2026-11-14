from types import SimpleNamespace
from unittest.mock import Mock

from window_view import run_window


def test_disconnected_camera_does_not_crash_preview(monkeypatch):
    keys = iter((-1, ord('q')))
    monkeypatch.setattr('window_view.cv2.imshow', lambda *a: None)
    monkeypatch.setattr('window_view.cv2.waitKey', lambda *a: next(keys))
    monkeypatch.setattr('window_view.cv2.destroyAllWindows', lambda: None)
    state = SimpleNamespace(capture=None, stop=Mock(), get_jpeg=lambda: None,
                            _thread=SimpleNamespace(is_alive=lambda: True),
                            get_status=lambda: dict(error='', alert='', fps=0, model='test'))
    run_window(state, 'test')
    state.stop.assert_called_once()
