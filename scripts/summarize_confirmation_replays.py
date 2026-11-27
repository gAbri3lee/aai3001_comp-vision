"""Summarise replay_alerts.py runs: spoken alerts with and without K-of-N frame confirmation.

    python scripts/summarize_confirmation_replays.py artifacts/replay

Files are named <config>__<video>.jsonl. For each config: spoken alerts, alerts per minute, alerts started by a
detection seen in one frame only (a single-frame flicker: no same-label box in the frames just before and after),
and the median delay of each video's first alert per class compared with the first config.
"""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from golden import iou


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def label_of(row):
    return (row.get('selected') or {}).get('cls') or row['speak']


def flicker(rows, i, label):
    """True if the spoken label's box in frame i has no same-label overlap in frames i-1 and i+1."""
    boxes = [d['box_xyxy'] for d in rows[i]['detections'] if d['class'] == label and d.get('gated')]
    if not boxes:
        return False
    neighbours = [rows[j] for j in (i - 1, i + 1) if 0 <= j < len(rows)]
    return not any(iou(b, d['box_xyxy']) >= .2 for b in boxes for r in neighbours
                   for d in r['detections'] if d['class'] == label)


def main(folder):
    runs = defaultdict(dict)
    for path in sorted(Path(folder).glob('*__*.jsonl')):
        config, video = path.stem.split('__', 1)
        runs[config][video] = load(path)
    configs = list(runs)
    first = {}
    print(f"{'config':<12} {'spoken':>7} {'per min':>8} {'single-frame':>13} {'first-alert delay vs ' + configs[0]:>30}")
    for config in configs:
        spoken = flick = 0
        seconds = 0.
        firsts = {}
        for video, rows in runs[config].items():
            seconds += rows[-1]['time_s'] if rows else 0
            for i, r in enumerate(rows):
                if r['speak']:
                    spoken += 1
                    label = label_of(r)
                    flick += flicker(rows, i, label)
                    firsts.setdefault((video, label), r['time_s'])
        first[config] = firsts
        delays = [firsts[k] - first[configs[0]][k] for k in firsts if k in first[configs[0]]]
        lost = [k for k in first[configs[0]] if k not in firsts]
        delay = (f"median {statistics.median(delays) * 1000:.0f} ms, max {max(delays) * 1000:.0f} ms" if delays else "-")
        if lost:
            detail = [f"{v}: {l} (gated in {sum(any(d['class'] == l and d.get('gated') for d in r['detections']) for r in runs[config][v])} frames)"
                      for v, l in lost]
            delay += f"; never spoken: {detail}"
        print(f"{config:<12} {spoken:>7} {spoken / max(seconds / 60, 1e-9):>8.1f} {flick:>13} {delay:>30}")
    print(f"\n{len(runs[configs[0]])} videos, {seconds:.0f} s of footage per config")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'artifacts/replay')
