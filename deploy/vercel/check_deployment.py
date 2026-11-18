"""Golden-set parity + round-trip latency for a deployed (or local) /api endpoint.

    python deploy/vercel/check_deployment.py https://<your-app>.vercel.app
    python deploy/vercel/check_deployment.py http://127.0.0.1:3000
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import requests

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'src'))
from golden import compare_outputs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_url')
    parser.add_argument('--reference', type=Path, default=root / 'golden/expected/local_yolov8n_four_class.json')
    parser.add_argument('--images', type=Path, default=root / 'golden/images')
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    health = requests.get(base + '/api/health', timeout=60).json()
    print('health:', health.get('status'), health.get('model'))
    outputs, round_trips = {}, []
    for path in sorted(args.images.glob('*.jpg')):
        start = time.perf_counter()
        response = requests.post(base + '/api/detect', data=path.read_bytes(),
                                 headers={'Content-Type': 'image/jpeg'}, timeout=60)
        round_trips.append(1000 * (time.perf_counter() - start))
        response.raise_for_status()
        outputs[path.name] = response.json()
    result = compare_outputs(json.loads(args.reference.read_text()), outputs)
    ordered = sorted(round_trips)
    result['round_trip_ms'] = {'first': round(round_trips[0], 1), 'median': round(statistics.median(round_trips), 1),
                               'p95': round(ordered[int(.95 * (len(ordered) - 1))], 1)}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
