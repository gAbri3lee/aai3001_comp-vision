"""Which supplementary COCO model file the app loads: INT8 ONNX on CPU, else the full-precision .pt."""
import argparse
import builtins
from unittest.mock import patch

from config import add_detector_arguments, detector_settings, load_config, resolve_path
from scene_detector import general_weights

SPEC = load_config()['general']
INT8, PT = resolve_path(SPEC['file']), resolve_path(SPEC['fp32_file'])


def test_cpu_uses_the_shipped_int8_model():
    assert INT8.is_file(), 'models/general/yolov8n_int8.onnx is committed and must exist'
    assert general_weights(device='cpu') == INT8


def test_other_devices_use_the_full_precision_model():
    assert general_weights(device='mps') == PT
    assert general_weights(device='0') == PT


def test_missing_int8_file_falls_back(tmp_path):
    config = dict(load_config(), general=dict(SPEC, file=str(tmp_path / 'absent.onnx')))
    with patch('scene_detector.load_config', return_value=config):
        assert general_weights(device='cpu') == PT


def test_missing_onnx_runtime_falls_back():
    real_import = builtins.__import__

    def without_onnxruntime(name, *args, **kwargs):
        if name == 'onnxruntime':
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=without_onnxruntime):
        assert general_weights(device='cpu') == PT


def test_flag_overrides_the_default():
    parser = argparse.ArgumentParser()
    add_detector_arguments(parser)
    assert detector_settings(parser.parse_args([]))['general_weights'] is None
    settings = detector_settings(parser.parse_args(['--general-model', 'models/general/yolov8n.pt']))
    assert settings['general_weights'] == PT
    assert general_weights(settings['general_weights'], 'cpu') == PT


def test_int8_model_loads_with_the_coco_names():
    from scene_detector import load_general_model
    model = load_general_model(device='cpu')
    assert len(model.names) == 80 and model.names[0] == 'person'
