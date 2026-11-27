"""python scripts/summarize_class_threshold_replays.py artifacts/replay_thresholds

Per-class effect of lower Stair / Wet Floor Sign thresholds and 2-of-3 confirmation on the demo replays.

Videos are not frame-labelled, so this uses which hazards each clip is known to contain (demo/README.md):
detections of a class in a clip that contains it ~ hits; in a clip that does not ~ likely false alarms.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

CONTAINS = {   # from demo/README.md and the clip titles
    'Stair': {'Walk_to_the_subway_stairs', 'Passengers_descending_the_central_staircase',
              'Stairs_and_pedestrians_-_field_sequence'},
    'Wet Floor Sign': {'Wet_floor_sign_-_field_sequence'},
}

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/replay_thresholds")
runs = defaultdict(dict)
for path in sorted(folder.glob('*__*.jsonl')):
    config, video = path.stem.split('__', 1)
    runs[config][video] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def spoken_label(row):
    return (row.get('selected') or {}).get('cls')


for cls, targets in CONTAINS.items():
    print(f'\n== {cls} ==  (clips that contain it: {len(targets)})')
    print(f"{'config':<14}{'frames w/ box (in clips with it)':>34}{'gated':>8}{'spoken':>8}"
          f"{'| frames w/ box (other clips)':>30}{'spoken':>8}")
    for config, videos in runs.items():
        hit_frames = hit_total = gated = spoken_in = fp_frames = spoken_out = 0
        for video, rows in videos.items():
            inside = video in targets
            for r in rows:
                dets = [d for d in r['detections'] if d['class'] == cls and d.get('source') != 'general']
                spoke = bool(r['speak']) and spoken_label(r) == cls
                if inside:
                    hit_total += 1
                    hit_frames += bool(dets)
                    gated += any(d.get('gated') for d in dets)
                    spoken_in += spoke
                else:
                    fp_frames += bool(dets)
                    spoken_out += spoke
        print(f"{config:<14}{f'{hit_frames}/{hit_total} ({hit_frames / hit_total:.0%})':>34}{gated:>8}{spoken_in:>8}"
              f"{fp_frames:>30}{spoken_out:>8}")

print('\nPer-clip frames with a box (trained model):')
for cls, targets in CONTAINS.items():
    for video in sorted(targets):
        cells = []
        for config, videos in runs.items():
            rows = videos.get(video, [])
            n = sum(any(d['class'] == cls and d.get('source') != 'general' for d in r['detections']) for r in rows)
            cells.append(f'{config}={n}/{len(rows)}')
        print(f'  {cls:<15}{video:<46}' + '  '.join(cells))

print('\nFalse-alarm detail (spoken Stair / Wet Floor Sign in clips without it):')
for config, videos in runs.items():
    for video, rows in videos.items():
        for r in rows:
            label = spoken_label(r)
            if r['speak'] and label in CONTAINS and video not in CONTAINS[label]:
                conf = max((d['confidence'] for d in r['detections'] if d['class'] == label), default=0)
                print(f'  {config:<14}{video:<46}t={r["time_s"]:>6.2f}s  {r["speak"]!r}  conf={conf:.2f}')
