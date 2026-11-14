import pytest
from webapp import decode_upload, UploadError


def test_garbage_never_reaches_optional_image_codec(monkeypatch):
    monkeypatch.setattr('PIL.Image.open', lambda *a, **kw: pytest.fail('non-image reached codec'))
    with pytest.raises(UploadError) as error:
        decode_upload(b'not an image')
    assert error.value.status == 415


def test_missing_optional_codec_does_not_crash_upload(monkeypatch):
    def broken(*a, **kw):
        raise ModuleNotFoundError('pi_heif')
    monkeypatch.setattr('PIL.Image.open', broken)
    with pytest.raises(UploadError):
        decode_upload(b'\xff\xd8\xffbroken')
