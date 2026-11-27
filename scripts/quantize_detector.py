"""Quantise the verified FP32 ONNX detector to INT8 and check it against FP32 before anyone uses it.

    python scripts/quantize_detector.py                       # calibrate, quantise, compare, write a report
    python scripts/quantize_detector.py --calib data/v4_four_class_clean/train/images --calib-count 300

1. Calibration: INT8 needs the typical range of every layer's values. It runs the FP32 model on calibration
   images (training images when you have them; otherwise frames from demo/), preprocessed exactly as the app does.
2. Quantisation: static, QDQ format, INT8 weights per channel. The last layers of the detection head (box
   decoding, the final conv of each box/class branch, DFL) stay in FP32: quantising them breaks the outputs
   (0 of 20 detections kept in testing), while the rest of the network quantises safely.
3. Checks against FP32 on held-out images: golden parity (the app's strict deployment check), a looser
   same-class / IoU >= 0.5 agreement, confidence drift and CPU latency. The report says whether INT8 is safe to try.

Adopting INT8 in the app still needs the validation AP check (mAP and safety recall) on the cleaned split.
This script never edits visionaid_config.json.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import add_detector_arguments, detector_settings, resolve_path
from golden import compare_outputs, iou

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
VIDEO_SUFFIXES = ('.mp4', '.mov', '.avi', '.mkv')
DEFAULT_FP32 = 'deploy/vercel/model/yolov8n_v4_four_class.onnx'
DEFAULT_INT8 = 'models/local/yolov8n_v4_four_class_int8.onnx'


def calibration_frames(sources, count, every_nth_video_frame=15):
    """Collect up to `count` BGR frames from image folders and videos (sorted, deterministic)."""
    import cv2
    frames = []
    for source in sources:
        source = resolve_path(source)
        files = sorted(source.rglob('*')) if source.is_dir() else [source]
        for path in files:
            if len(frames) >= count:
                return frames
            suffix = path.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                image = cv2.imread(str(path))
                if image is not None:
                    frames.append(image)
            elif suffix in VIDEO_SUFFIXES:
                cap = cv2.VideoCapture(str(path))
                index = 0
                while len(frames) < count:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if index % every_nth_video_frame == 0:
                        frames.append(frame)
                    index += 1
                cap.release()
    return frames


def preprocess(frame_bgr, size):
    """Exactly the app's ONNX preprocessing (src/onnx_detector.py)."""
    import numpy as np
    from ultralytics.data.augment import LetterBox
    frame = LetterBox(new_shape=(size, size), auto=False, stride=32)(image=frame_bgr)
    return np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def quantize(fp32, int8, frames, size):
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.name = onnx.load(str(fp32), load_external_data=False).graph.input[0].name
            self.items = iter(frames)

        def get_next(self):
            frame = next(self.items, None)
            return None if frame is None else {self.name: preprocess(frame, size)}

    int8.parent.mkdir(parents=True, exist_ok=True)
    prepared = int8.with_suffix('.prep.onnx')
    quant_pre_process(str(fp32), str(prepared), skip_symbolic_shape=True)
    model = onnx.load(str(prepared))
    # Keep in FP32: the box/score decoding (every non-Conv node of the head, model.22), the final 1x1 conv of each
    # box and class branch, and the DFL conv. Quantising those wrecks the outputs; the earlier head convs are safe.
    head = [n for n in model.graph.node if '/model.22/' in n.name]
    keep = [n.name for n in head if n.op_type != 'Conv' or n.name.split('/')[-2].endswith('.2') or '/dfl/' in n.name]
    quantize_static(str(prepared), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=keep)
    prepared.unlink(missing_ok=True)
    # Keep the export metadata (task, class names) that the app's ONNX adapter validates.
    source, target = onnx.load(str(fp32), load_external_data=False), onnx.load(str(int8))
    del target.metadata_props[:]
    target.metadata_props.extend(source.metadata_props)
    onnx.save(target, str(int8))
    return len(keep)


def run(weights, images, threads, config_path):
    """Detections (app thresholds) and per-image latency for one ONNX file, via the app's own adapter."""
    import cv2
    from onnx_detector import OnnxDetector
    parser = argparse.ArgumentParser()
    add_detector_arguments(parser)
    argv = ['--model', str(weights)] + (['--config', config_path] if config_path else [])
    settings = dict(detector_settings(parser.parse_args(argv)), onnx_threads=threads)
    model = OnnxDetector(settings)
    outputs, times = {}, []
    for path in images:
        frame = cv2.imread(str(path))
        model.predict(frame)                                   # warm-up / cache
        batch = model.predict(frame)
        times.append(batch.latency_ms)
        outputs[path.name] = {'image_size': [frame.shape[1], frame.shape[0]],
                              'detections': [d.to_dict() for d in batch.detections]}
    return outputs, times


def agreement(reference, outputs, min_iou=.5):
    """Looser than golden parity: same class and IoU >= min_iou counts as the same detection."""
    matched = total = extra = 0
    for name, item in reference.items():
        remaining = list(outputs.get(name, {}).get('detections', []))
        for d in item['detections']:
            total += 1
            same = [o for o in remaining if o['class_id'] == d['class_id']]
            best = max(same, key=lambda o: iou(o['box_xyxy'], d['box_xyxy']), default=None)
            if best is not None and iou(best['box_xyxy'], d['box_xyxy']) >= min_iou:
                matched += 1
                remaining.remove(best)
        extra += len(remaining)
    return {'matched': matched, 'total': total, 'missed_share': 1 - matched / total if total else 0., 'extra': extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fp32', default=DEFAULT_FP32)
    parser.add_argument('--output', default=DEFAULT_INT8)
    parser.add_argument('--calib', nargs='+', default=None,
                        help='Image folders or videos for calibration (default: the cleaned train split if present, else demo/)')
    parser.add_argument('--calib-count', type=int, default=200)
    parser.add_argument('--eval-images', default='golden/images', help='Held-out images for the FP32-vs-INT8 checks')
    parser.add_argument('--threads', type=int, default=4, help='ONNX Runtime CPU threads for the latency check')
    parser.add_argument('--config', default=None)
    parser.add_argument('--report', default='artifacts/int8-quantization.json')
    parser.add_argument('--skip-quantize', action='store_true', help='Only re-run the checks on an existing INT8 file')
    args = parser.parse_args()

    fp32, int8 = resolve_path(args.fp32), resolve_path(args.output)
    if not fp32.is_file():
        parser.error(f'Missing FP32 ONNX: {fp32}')
    size = 640
    if not args.skip_quantize:
        train = resolve_path('data/v4_four_class_clean/train/images')
        sources = args.calib or ([train] if train.is_dir() else ['demo'])
        frames = calibration_frames(sources, args.calib_count)
        if not frames:
            parser.error(f'No calibration images found in {sources}')
        print(f'calibrating on {len(frames)} frames from {[str(s) for s in sources]}')
        excluded = quantize(fp32, int8, frames, size)
        print(f'wrote {int8} ({excluded} detection-head nodes kept in FP32)')

    images = sorted(p for p in resolve_path(args.eval_images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    ref, t_fp32 = run(fp32, images, args.threads, args.config)
    out, t_int8 = run(int8, images, args.threads, args.config)
    golden = compare_outputs(ref, out)
    loose = agreement(ref, out)
    report = {
        'fp32': str(args.fp32), 'int8': str(args.output), 'eval_images': len(images), 'threads': args.threads,
        'size_mb': {'fp32': round(fp32.stat().st_size / 2**20, 2), 'int8': round(int8.stat().st_size / 2**20, 2)},
        'median_ms': {'fp32': round(statistics.median(t_fp32), 1), 'int8': round(statistics.median(t_int8), 1)},
        'speedup': round(statistics.median(t_fp32) / statistics.median(t_int8), 2),
        'golden_parity_vs_fp32': golden,
        'agreement_vs_fp32_iou50': loose,
        'note': ('Agreement is measured against the FP32 model on held-out images, not against labels. '
                 'Adopt INT8 only after the validation mAP / safety-recall check on the cleaned split.'),
    }
    print(json.dumps(report, indent=2))
    out_path = resolve_path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print('report:', out_path)


if __name__ == '__main__':
    main()
