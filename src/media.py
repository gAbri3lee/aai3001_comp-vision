"""Uploaded media for the demo: batch limits, display names and honest scores.

Uploaded files are untrusted. A client file name is only ever shown on the page,
after cleaning, and is never used in a path.
"""
import math
import unicodedata

from config import CLASS_NAMES

# Decision V4: up to 20 images per batch, each a JPEG or PNG of at most 10 MB.
MAX_BATCH = 20
MAX_DISPLAY_SIDE = 1280   # annotated images sent back to the page are at most this big
NAME_LENGTH = 80


def display_name(filename, fallback):
    """A name to show on the page: the last path part of the client's file name,
    without control characters and cut to a sensible length. Never used as a path."""
    name = (filename or '').replace('\\', '/').rsplit('/', 1)[-1]
    name = ''.join(ch for ch in name if unicodedata.category(ch)[0] != 'C').strip()
    if len(name) > NAME_LENGTH:
        name = name[:NAME_LENGTH - 3] + '...'
    return name or fallback


def recognition_dicts(hazards):
    """Recognition view (decision V1): every trained detection that passed its class
    confidence threshold, with no distance, side or general-model filter. Highest
    confidence first."""
    trained = sorted((h for h in hazards if h.source == 'trained'), key=lambda h: -h.confidence)
    return [{'class_id': h.class_id, 'class': h.label, 'confidence': h.confidence,
             'box_xyxy': list(h.xyxy), 'position': h.position, 'size': h.size} for h in trained]


def expected_result(recognition, expected):
    """Decision V5: recognised when any detection of the expected class exists."""
    scores = [d['confidence'] for d in recognition if d['class'] == expected]
    return {'class': expected, 'recognised': bool(scores),
            'confidence': max(scores) if scores else None}


def batch_summary(results, backend, model, expected):
    """For example "Stair: 8 of 10 recognised, mean confidence 0.71". The mean is
    over the best expected-class confidence of each recognised image."""
    runs = [run for entry in results for run in entry.get('runs', ())
            if run.get('mode') == backend and 'error' not in run]
    scores = [run['expected']['confidence'] for run in runs if run['expected']['recognised']]
    mean = sum(scores) / len(scores) if scores else None
    skipped = sum(1 for entry in results if 'error' in entry) + sum(
        1 for entry in results for run in entry.get('runs', ()) if run.get('mode') == backend and 'error' in run)
    text = f'{expected}: {len(scores)} of {len(runs)} recognised'
    if mean is not None:
        text += f', mean confidence {mean:.2f}'
    return {'backend': backend, 'model': model, 'class': expected, 'recognised': len(scores),
            'total': len(runs), 'skipped': skipped,
            'mean_confidence': None if mean is None else round(mean, 2), 'text': text}


def _three_places(value):
    # Truncated, so a rounded figure never overstates the recorded result.
    return f'{math.floor(value * 1000) / 1000:.3f}'


def accuracy_note(config):
    """The caption shown next to every score, from the recorded test results."""
    test_images = config['dataset']['split_images']['test']
    return (f"This is a check on these images only. Tested accuracy on {test_images} test images: "
            f"local mAP50 {_three_places(config['local']['expected']['test_map50'])}, "
            f"cloud mAP50 {_three_places(config['cloud']['expected']['test_map50'])}.")


def valid_expected(value):
    if value in (None, ''):
        return None
    if value not in CLASS_NAMES:
        raise ValueError('expected must be one of: ' + ', '.join(CLASS_NAMES))
    return value
