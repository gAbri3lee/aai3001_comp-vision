"""Replay a video or image folder through the web app's alert pipeline, without audio.

Run mode uses the same pieces as webapp.py: the trained detector plus the general
COCO model (SceneDetector), then AlertPolicy (alert gate, ranking, cooldown and the
spoken-hazard choice). Every frame is processed in order on a simulated clock
(frame index / FPS), so local and cloud runs line up frame by frame. Speech is never
played. Output is JSON lines: frame index, time, detections with gate fields, gated
phrases, banner, selected hazard, phrase that would be spoken, timings.

    python scripts/replay_alerts.py --source artifacts/golden_clip.mp4 --output artifacts/replay_local.jsonl
    python scripts/replay_alerts.py --source artifacts/golden_clip.mp4 --backend cloud \\
        --endpoint http://127.0.0.1:9100 --output artifacts/replay_cloud.jsonl
    python scripts/replay_alerts.py --compare artifacts/replay_local.jsonl artifacts/replay_cloud.jsonl

Compare mode lists frames where the banner or spoken phrase differ, per-class
detection counts and mean latency. It also re-runs the gate on each file's recorded
detections: the gate is deterministic, so any mismatch there is a gate bug. Differences
between backends are expected when their detections differ (different models and
thresholds).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import add_detector_arguments, detector_settings
from decision_engine import AlertPolicy, assess, phrase_for
from detection import Detection

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def frames(source, stride, fps):
    """Yield (index, seconds, frame) from a video file or an image folder."""
    import cv2
    source = Path(source)
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for i, path in enumerate(paths[::stride]):
            yield i * stride, i * stride / fps, cv2.imread(str(path))
        return
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f'Could not open {source}')
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                yield index, index / video_fps, frame
            index += 1
    finally:
        cap.release()


def run(args):
    from scene_detector import SceneDetector
    settings = detector_settings(args)
    detector = SceneDetector(settings)
    clock = {'now': 0.}
    policy = AlertPolicy(settings['alert_gate'], settings['severity'], clock=lambda: clock['now'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(args.output, 'w') as out:
        for index, seconds, frame in frames(args.source, args.stride, args.fps):
            clock['now'] = seconds
            batch = detector.predict(frame)
            height, width = frame.shape[:2]
            result = policy.process(batch.detections, (width, height), now=seconds)
            hazards = sorted(result.hazards, key=lambda h: h.index)
            out.write(json.dumps(dict(
                frame=index, time_s=round(seconds, 3), width=width, height=height,
                backend=args.backend, model=batch.model, latency_ms=batch.latency_ms,
                round_trip_ms=batch.round_trip_ms,
                detections=[h.to_dict() for h in hazards], alerts=result.alerts, banner=result.banner,
                selected=None if result.selected is None else dict(
                    cls=result.selected.label, position=result.selected.position, band=result.selected.band,
                    phrase=phrase_for(result.selected)),
                speak=result.speak)) + '\n')
            count += 1
    print(f'{count} frames written to {args.output}')


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def regate_mismatches(records, gate, severity):
    """Frames whose recorded gate output differs from re-running the gate on their detections."""
    bad = []
    for record in records:
        detections = [Detection(d['class_id'], d['class'], d['confidence'], tuple(d['box_xyxy']))
                      for d in record['detections']]
        again = sorted(assess(detections, (record['width'], record['height']), gate, severity),
                       key=lambda h: h.index)
        recorded = [(d['class'], d['position'], d['band'], d['gated'], d['reason']) for d in record['detections']]
        if recorded != [(h.label, h.position, h.band, h.gated, h.reason) for h in again]:
            bad.append(record['frame'])
    return bad


def compare(args):
    settings = detector_settings(args)
    a, b = load(args.compare[0]), load(args.compare[1])
    name_a, name_b = a[0]['backend'] if a else 'a', b[0]['backend'] if b else 'b'
    by_frame_b = {r['frame']: r for r in b}
    common = [r for r in a if r['frame'] in by_frame_b]
    lines = [f'Frames: {name_a} {len(a)}, {name_b} {len(b)}, compared {len(common)}', '']

    def counts(records, gated_only):
        return Counter(d['class'] + (' (general)' if d['source'] == 'general' and d['class'] != 'Object' else '')
                       for r in records for d in r['detections'] if d['gated'] or not gated_only)
    for title, gated_only in (('Detections per class (all)', False), ('Gated detections per class', True)):
        ca, cb = counts(common, gated_only), counts([by_frame_b[r['frame']] for r in common], gated_only)
        lines += [f'{title}:', f'| Class | {name_a} | {name_b} |', '|---|---:|---:|']
        lines += [f'| {k} | {ca[k]} | {cb[k]} |' for k in sorted(set(ca) | set(cb))]
        lines.append('')

    def timing(records):
        # latency_ms: trained model (server time for cloud) plus the local general model.
        # round_trip_ms: cloud only, trained-model HTTP round trip from the client.
        parts = []
        for field in ('latency_ms', 'round_trip_ms'):
            values = [r[field] for r in records if r[field] is not None]
            if values:
                parts.append(f'{field} mean {statistics.mean(values):.0f}, median {statistics.median(values):.0f}')
        return '; '.join(parts) or '-'
    lines += [f'Timing ({name_a}): {timing(a)}', f'Timing ({name_b}): {timing(b)}',
              'latency_ms includes the local general model; round_trip_ms is the cloud HTTP request alone.', '']

    banner_diff = [r for r in common if r['banner'] != by_frame_b[r['frame']]['banner']]
    speak_diff = [r for r in common if r['speak'] != by_frame_b[r['frame']]['speak']]
    same_detections_diff = []
    for r in banner_diff:
        other = by_frame_b[r['frame']]
        key = lambda rec: sorted((d['class'], d['position'], d['band']) for d in rec['detections'] if d['gated'])
        if key(r) == key(other):
            same_detections_diff.append(r['frame'])
    lines.append(f'Banner differs on {len(banner_diff)} of {len(common)} frames; '
                 f'spoken phrase differs on {len(speak_diff)} frames.')
    lines.append(f'Banner differs although both gated the same (class, position, band) set: '
                 f'{len(same_detections_diff)} frames {same_detections_diff[:10]}')
    for name, records in ((name_a, a), (name_b, b)):
        bad = regate_mismatches(records, settings['alert_gate'], settings['severity'])
        lines.append(f'Gate re-check on {name}: {"all frames reproduce" if not bad else f"MISMATCH on frames {bad[:10]}"}')
    lines += ['', f'| Frame | Time | {name_a} banner | {name_b} banner | {name_a} spoken | {name_b} spoken |',
              '|---:|---:|---|---|---|---|']
    for r in sorted({r['frame']: r for r in banner_diff + speak_diff}.values(), key=lambda r: r['frame']):
        o = by_frame_b[r['frame']]
        lines.append(f"| {r['frame']} | {r['time_s']:.1f} | {r['banner'] or '-'} | {o['banner'] or '-'} | "
                     f"{r['speak'] or '-'} | {o['speak'] or '-'} |")
    text = '\n'.join(lines) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_detector_arguments(parser)
    parser.add_argument('--source', help='Video file or image folder')
    parser.add_argument('--output', type=Path, help='JSONL output (run) or markdown summary (compare)')
    parser.add_argument('--stride', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--fps', type=float, default=10.0, help='Simulated frame rate for image folders')
    parser.add_argument('--no-tts', action='store_true', help='Accepted for symmetry; replay never plays audio')
    parser.add_argument('--compare', nargs=2, type=Path, metavar=('A.jsonl', 'B.jsonl'))
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be at least 1')
    if args.compare:
        compare(args)
    else:
        if not args.source or not args.output:
            parser.error('--source and --output are required unless --compare is given')
        run(args)


if __name__ == '__main__':
    main()
