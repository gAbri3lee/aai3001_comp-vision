import os

from media import TempStore, TEMP_PREFIX


def test_stale_cleanup_preserves_live_process_and_removes_dead_process(tmp_path, monkeypatch):
    monkeypatch.setattr('tempfile.gettempdir', lambda: str(tmp_path))
    live = tmp_path / f'{TEMP_PREFIX}{os.getpid()}-live'
    dead = tmp_path / f'{TEMP_PREFIX}2147483647-dead'
    unknown = tmp_path / f'{TEMP_PREFIX}not-a-pid'
    for folder in (live, dead, unknown):
        folder.mkdir()
        (folder / 'uploaded.mp4').write_bytes(b'test')
    TempStore.remove_stale()
    assert live.exists() and unknown.exists()
    assert not dead.exists()
