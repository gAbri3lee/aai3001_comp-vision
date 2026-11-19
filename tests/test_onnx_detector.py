"""Real ONNX Runtime contract and non-square coordinate regressions."""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from config import add_detector_arguments, detector_settings
from onnx_detector import OnnxDetector


def exported_fixture(tmp_path, names=None):
    path = tmp_path / 'fixture.onnx'
    output = np.zeros((1, 8, 1), dtype=np.float32)
    output[0, :4, 0] = [320, 320, 320, 160]
    output[0, 5, 0] = .9  # People class at channel 4+1.
    node = helper.make_node('Constant', [], ['output0'], value=numpy_helper.from_array(output))
    graph = helper.make_graph([node], 'coordinate fixture',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1, 8, 1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 10
    names = names or ['Escalator', 'People', 'Stair', 'Wet Floor Sign']
    helper.set_model_props(model, {'task': 'detect', 'names': repr(dict(enumerate(names)))})
    onnx.save(model, path)
    parser = argparse.ArgumentParser()
    add_detector_arguments(parser)
    return detector_settings(parser.parse_args(['--model', str(path)]))


def test_onnx_runtime_restores_original_non_square_coordinates(tmp_path):
    model = OnnxDetector(exported_fixture(tmp_path))
    batch = model.predict(np.zeros((160, 320, 3), dtype=np.uint8))
    assert len(batch.detections) == 1
    detection = batch.detections[0]
    assert detection.class_id == 1
    assert detection.label == 'People'
    assert detection.box_xyxy == pytest.approx((80, 40, 240, 120))
    assert detection.confidence == pytest.approx(.9)


def test_onnx_runtime_rejects_reordered_classes(tmp_path):
    settings = exported_fixture(tmp_path, ['People', 'Escalator', 'Stair', 'Wet Floor Sign'])
    with pytest.raises(ValueError, match='classes'):
        OnnxDetector(settings)


def test_onnx_runtime_rejects_input_size_mismatch(tmp_path):
    settings = exported_fixture(tmp_path)
    settings['input_size'] = 320
    with pytest.raises(ValueError, match='static batch-one'):
        OnnxDetector(settings)
