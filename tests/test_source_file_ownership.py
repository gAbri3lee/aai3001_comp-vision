"""Windows Replay/camera cancellation must never delete a live decoder's file."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from webapp import PipelineState


def test_cancelling_pending_replay_preserves_active_file_until_loop_retires_it(tmp_path):
    path=tmp_path/'uploaded.mp4';path.write_bytes(b'fixture')
    video=dict(path=path,temporary=True)
    state=PipelineState.__new__(PipelineState)
    state._lock=threading.Lock();state._source_io_lock=threading.RLock()
    state._generation=0;state.video=video
    replay=SimpleNamespace(release=Mock())
    state._pending=(replay,video)
    state.store=SimpleNamespace(delete=Mock(side_effect=PermissionError('active decoder still owns file')))
    state._switch(None,None)
    replay.release.assert_called_once()
    state.store.delete.assert_not_called()
    assert state._pending==(None,None) and state.view=='alert'


def test_replacing_unrelated_pending_upload_still_deletes_only_the_abandoned_file(tmp_path):
    active=dict(path=tmp_path/'active.mp4',temporary=True)
    pending=dict(path=tmp_path/'abandoned.mp4',temporary=True)
    state=PipelineState.__new__(PipelineState)
    state._lock=threading.Lock();state._source_io_lock=threading.RLock();state._generation=0;state.video=active
    capture=SimpleNamespace(release=Mock());state._pending=(capture,pending)
    state.store=SimpleNamespace(delete=Mock())
    state._switch(None,None)
    capture.release.assert_called_once();state.store.delete.assert_called_once_with(pending['path'])
