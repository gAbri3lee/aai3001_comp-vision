"""Four-class deployment contract, with explicit historical config support."""
import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ('Escalator', 'People', 'Stair', 'Wet Floor Sign')
LEGACY_CLASS_NAMES = ('Escalator', 'People', 'Pillar', 'Stair', 'Wet Floor Sign')


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path=None):
    config = json.loads(resolve_path(path or 'visionaid_config.json').read_text())
    names = config['classes']
    if names not in (list(CLASS_NAMES), list(LEGACY_CLASS_NAMES)):
        raise ValueError('Class IDs must match the four-class contract or explicit historical five-class order')
    for backend in ('local', 'cloud'):
        spec = config[backend]
        if set(spec['class_thresholds']) != set(names):
            raise ValueError(f'{backend}: thresholds must cover exactly the configured classes')
        source = spec.get('source_classes', names)
        if source not in (names, list(LEGACY_CLASS_NAMES)) or not set(names).issubset(source):
            raise ValueError(f'{backend}: unsupported checkpoint class mapping')
        if 'legacy_service_classes' in spec and spec['legacy_service_classes'] != list(LEGACY_CLASS_NAMES):
            raise ValueError('Unsupported legacy service class order')
        for value in spec['class_thresholds'].values():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Confidence thresholds must be finite and between 0 and 1')
    if set(config['severity']) != set(names):
        raise ValueError('Severity must cover exactly the configured classes')
    validate_alert_gate(config['alert_gate'], names)
    return config


GATE_LABELS = set(CLASS_NAMES) | {'Object'}


def _fraction(value, name, upper=1.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= upper:
        raise ValueError(f'alert_gate.{name} must be a finite number in (0, {upper}]')


def validate_alert_gate(gate, names=CLASS_NAMES):
    gate_labels = set(names) | {'Object'}
    if not isinstance(gate.get('enabled'), bool):
        raise ValueError('alert_gate.enabled must be true or false')
    if gate.get('size_metric') != 'box_height_over_frame_height':
        raise ValueError('alert_gate.size_metric must be box_height_over_frame_height')
    low, high = gate['centre_third']
    _fraction(low, 'centre_third'); _fraction(high, 'centre_third')
    if not low < high:
        raise ValueError('alert_gate.centre_third must be [low, high] with low < high')
    _fraction(gate['side_multiplier'], 'side_multiplier', upper=10.0)
    for table in ('min_size', 'near_size'):
        if set(gate[table]) != gate_labels:
            raise ValueError(f'alert_gate.{table} must cover the configured classes and Object')
        for label, value in gate[table].items():
            _fraction(value, f'{table}.{label}')
    for label in gate_labels:
        if gate['near_size'][label] < gate['min_size'][label]:
            raise ValueError(f'alert_gate.near_size.{label} must not be below min_size')
    _fraction(gate['object_min_confidence'], 'object_min_confidence')
    if set(gate['cooldown_seconds']) != {'trained', 'Object'}:
        raise ValueError('alert_gate.cooldown_seconds needs trained and Object')
    for key, value in gate['cooldown_seconds'].items():
        _fraction(value, f'cooldown_seconds.{key}', upper=600.0)
    _fraction(gate['proximity_jump_to_realert'], 'proximity_jump_to_realert')
    for label, value in gate.get('known_heights_m', {}).items():
        if label not in names:
            raise ValueError(f'alert_gate.known_heights_m: unknown class {label}')
        _fraction(value, f'known_heights_m.{label}', upper=10.0)


def add_detector_arguments(parser):
    parser.add_argument('--config', default=None, help='Config path, relative to the project unless absolute')
    parser.add_argument('--backend', choices=['local', 'cloud'], default='local')
    parser.add_argument('--model', default=None, help='Override local weights; cloud weights are set on the server')
    parser.add_argument('--conf', type=float, default=None, help='Override all class thresholds (0-1)')
    parser.add_argument('--imgsz', type=int, default=None, help='Override local inference size')
    parser.add_argument('--device', default='cpu', help='Local inference device: cpu, mps, or CUDA index')
    parser.add_argument('--endpoint', default=None, help='Cloud service base URL; frames leave this device')
    parser.add_argument('--timeout', type=float, default=3.0, help='Cloud request deadline in seconds')
    parser.add_argument('--no-gate', action='store_true', help='Disable the alert gate: announce every detection (old behaviour)')
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
    gate = dict(config['alert_gate'])
    if getattr(args, 'no_gate', False):
        gate['enabled'] = False
    spec.update(classes=config['classes'], severity=config['severity'], backend=args.backend, alert_gate=gate,
                weights=resolve_path(args.model or spec['file']),
                confidence=args.conf if args.conf is not None else min(spec['class_thresholds'].values()),
                class_thresholds={n: args.conf if args.conf is not None else t for n, t in spec['class_thresholds'].items()},
                input_size=args.imgsz or spec['input_size'], device=args.device,
                endpoint=args.endpoint, timeout=args.timeout, max_frame_age=args.max_frame_age,
                confidence_override=args.conf)
    return spec
