"""Single source of truth for the trained five-class deployment."""
import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ('Escalator', 'People', 'Pillar', 'Stair', 'Wet Floor Sign')


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path=None):
    config = json.loads(resolve_path(path or 'visionaid_config.json').read_text())
    if config['classes'] != list(CLASS_NAMES):
        raise ValueError('Class IDs must remain 0 Escalator, 1 People, 2 Pillar, 3 Stair, 4 Wet Floor Sign')
    for backend in ('local', 'cloud'):
        spec = config[backend]
        if set(spec['class_thresholds']) != set(CLASS_NAMES):
            raise ValueError(f'{backend}: thresholds must cover exactly the five classes')
        for value in spec['class_thresholds'].values():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Confidence thresholds must be finite and between 0 and 1')
    if set(config['severity']) != set(CLASS_NAMES):
        raise ValueError('Severity must cover exactly the five classes')
    return config


def add_detector_arguments(parser):
    parser.add_argument('--config', default=None, help='Config path, relative to the project unless absolute')
    parser.add_argument('--backend', choices=['local', 'cloud'], default='local')
    parser.add_argument('--model', default=None, help='Override local weights; cloud weights are set on the server')
    parser.add_argument('--conf', type=float, default=None, help='Override all class thresholds (0-1)')
    parser.add_argument('--imgsz', type=int, default=None, help='Override local inference size')
    parser.add_argument('--device', default='cpu', help='Local inference device: cpu, mps, or CUDA index')
    parser.add_argument('--endpoint', default=None, help='Cloud service base URL; frames leave this device')
    parser.add_argument('--timeout', type=float, default=3.0, help='Cloud request deadline in seconds')
    parser.add_argument('--max-frame-age', type=float, default=3.0, help='Discard frames/results older than this many seconds')


def detector_settings(args):
    config = load_config(args.config)
    spec = dict(config[args.backend])
    if args.conf is not None and (not math.isfinite(args.conf) or not 0 <= args.conf <= 1):
        raise ValueError('--conf must be between 0 and 1')
    if args.imgsz is not None and args.imgsz <= 0:
        raise ValueError('--imgsz must be positive')
    if args.backend == 'cloud' and (args.model is not None or args.imgsz is not None):
        raise ValueError('--model and --imgsz apply locally; configure cloud weights and resolution on the server')
    if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.max_frame_age)):
        raise ValueError('Timeout and maximum frame age must be positive and finite')
    spec.update(classes=config['classes'], severity=config['severity'], backend=args.backend,
                weights=resolve_path(args.model or spec['file']),
                confidence=args.conf if args.conf is not None else min(spec['class_thresholds'].values()),
                class_thresholds={n: args.conf if args.conf is not None else t for n, t in spec['class_thresholds'].items()},
                input_size=args.imgsz or spec['input_size'], device=args.device,
                endpoint=args.endpoint, timeout=args.timeout, max_frame_age=args.max_frame_age,
                confidence_override=args.conf)
    return spec
