"""Optional FP32 CPU ONNX adapter with bounded thread use and shared YOLO NMS."""
import ast
import time

from detection import DetectionBatch, make_detection, validate_names


class OnnxDetector:
    def __init__(self, settings):
        import onnxruntime as ort
        self.settings = settings
        if settings['device'] != 'cpu':
            raise ValueError('The verified ONNX adapter supports --device cpu only')
        if not settings['weights'].is_file():
            raise FileNotFoundError(f"Missing ONNX weights: {settings['weights']}")
        options = ort.SessionOptions()
        threads = settings.get('onnx_threads', 4)
        if type(threads) is not int or not 1 <= threads <= 32:
            raise ValueError('ONNX thread count must be an integer from 1 to 32')
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(settings['weights']), options,
                                            providers=['CPUExecutionProvider'])
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get('task') != 'detect':
            raise ValueError('Expected a YOLO detection ONNX export')
        try:
            validate_names(ast.literal_eval(metadata['names']), settings['classes'])
        except (KeyError, SyntaxError, TypeError) as exc:
            raise ValueError('ONNX export lacks valid class metadata') from exc
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        size = settings['input_size']
        if len(inputs) != 1 or inputs[0].type != 'tensor(float)' or inputs[0].shape != [1, 3, size, size]:
            raise ValueError('Expected a static batch-one FP32 ONNX export matching --imgsz')
        if len(outputs) != 1 or len(outputs[0].shape) != 3 or outputs[0].shape[1] != 4 + len(settings['classes']):
            raise ValueError('Unexpected YOLOv8 ONNX output shape')
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def predict(self, frame_bgr):
        import numpy as np
        import torch
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils import ops
        from ultralytics.utils.nms import non_max_suppression
        start = time.perf_counter()
        spec = self.settings
        size = spec['input_size']
        frame = LetterBox(new_shape=(size, size), auto=False, stride=32)(image=frame_bgr)
        tensor = np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        boxes = non_max_suppression(torch.from_numpy(raw), conf_thres=spec['confidence'],
            iou_thres=spec['nms_iou'], max_det=300, nc=len(spec['classes']))[0]
        boxes[:, :4] = ops.scale_boxes(tensor.shape[2:], boxes[:, :4], frame_bgr.shape)
        height, width = frame_bgr.shape[:2]
        detections = []
        for x1, y1, x2, y2, confidence, cid in boxes.tolist():
            d = make_detection(cid, confidence, (x1, y1, x2, y2), spec['classes'], width, height)
            if d is not None and d.confidence >= spec['class_thresholds'][d.label]:
                detections.append(d)
        return DetectionBatch(tuple(sorted(detections, key=lambda d: -d.confidence)),
                              1000 * (time.perf_counter() - start), 'yolov8n')
