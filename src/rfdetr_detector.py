"""Server-only RF-DETR runtime; never imported by the local YOLO application."""
import time
from detection import DetectionBatch, make_detection, validate_names


class RFDetrDetector:
    def __init__(self, config, weights, device='cpu'):
        from pathlib import Path
        if not Path(weights).is_file():
            raise FileNotFoundError(f'Missing RF-DETR weights: {weights}')
        from rfdetr import RFDETRNano
        self.config = config
        self.spec = config['cloud']
        self.model = RFDETRNano(pretrain_weights=str(weights), num_classes=5,
                               resolution=self.spec['input_size'], device=device,
                               trust_checkpoint=True)
        # Only load the user-supplied trusted checkpoint, never an uploaded model.
        validate_names(self.model.class_names, config['classes'])

    def predict(self, image_rgb, confidence=None):
        start = time.perf_counter()
        output = self.model.predict(image_rgb.convert('RGB'), threshold=min(self.spec['class_thresholds'].values()) if confidence is None else confidence)
        width, height = image_rgb.size
        detections = []
        for cid, confidence, box in zip(output.class_id, output.confidence, output.xyxy):
            # RF-DETR has five foreground logits plus a final background slot.
            # Low-threshold AP runs can expose that slot; it is never a hazard.
            if int(cid) == len(self.config["classes"]):
                continue
            detection = make_detection(cid, confidence, box, self.config['classes'], width, height)
            if detection is not None and detection.confidence >= (self.spec['class_thresholds'][detection.label] if confidence is None else confidence):
                detections.append(detection)
        return DetectionBatch(tuple(sorted(detections, key=lambda d: d.confidence, reverse=True)),
                              1000 * (time.perf_counter() - start), 'rfdetr-nano')
