# Distance test sheet (tape measure)

Use this sheet to check the alert gate's distance bands with the real camera. The
app never measures distance. It uses box height as a stand-in, so this test shows
what each band means for your camera.

## Setup

1. Use the same laptop, camera and resolution you will demo with. Note them here:
   camera ______, resolution ______, date ______, tester ______.
2. Start the app with depth and speech as you will demo, for example:
   `python src/webapp.py --port 5001 --no-depth`
3. Open `http://127.0.0.1:5001/?debug=1`. The debug view draws ignored boxes and
   shows each detection's size (box height / frame height), band and, after
   calibration, the distance estimate.
4. Optional: calibrate first, so the debug panel shows an estimate for People and
   Wet Floor Sign. Stand a person (or place a sign) fully in frame at a measured
   3 m and run
   `python scripts/calibrate_distance.py --camera 0 --class People --distance 3`.
   Repeat at 2 m and 5 m; the app uses the median.
5. Mark distances on the floor from the camera lens: 1, 2, 3, 5 and 8 m. Keep the
   object in the centre third of the image unless the row says otherwise.
6. For each row, wait about 3 seconds, then read the values from the debug panel
   (or from `curl http://127.0.0.1:5001/status`).

Columns: **Detected** (yes or no), **Conf** (confidence), **Size** (box height
fraction), **Band** (far, mid, mid-close, near), **Spoken** (yes or no, from what
you hear), **Estimate** (distance estimate in metres if calibrated, otherwise blank).
A box cut off by the top or bottom edge has no estimate; note "cut off".

## People (1.7 m adult)

| Distance | Detected | Conf | Size | Band | Spoken | Estimate | Notes |
|---|---|---|---|---|---|---|---|
| 1 m | | | | | | | |
| 2 m | | | | | | | |
| 3 m | | | | | | | |
| 5 m | | | | | | | |
| 8 m | | | | | | | |
| 3 m, left third | | | | | | | |
| 3 m, right third | | | | | | | |

## Wet Floor Sign (about 0.65 m tall)

| Distance | Detected | Conf | Size | Band | Spoken | Estimate | Notes |
|---|---|---|---|---|---|---|---|
| 1 m | | | | | | | |
| 2 m | | | | | | | |
| 3 m | | | | | | | |
| 5 m | | | | | | | |
| 8 m | | | | | | | |

## Stair (distance to the first step)

| Distance | Detected | Conf | Size | Band | Spoken | Notes |
|---|---|---|---|---|---|---|
| 1 m | | | | | | |
| 2 m | | | | | | |
| 3 m | | | | | | |
| 5 m | | | | | | |
| 8 m | | | | | | |

## Escalator (distance to the entry)

| Distance | Detected | Conf | Size | Band | Spoken | Notes |
|---|---|---|---|---|---|---|
| 1 m | | | | | | |
| 2 m | | | | | | |
| 3 m | | | | | | |
| 5 m | | | | | | |
| 8 m | | | | | | |

## Pillar

Pillar is the weakest class (about 17 percent recall locally, 29 percent on the
cloud model). Misses are expected. Record them anyway.

| Distance | Detected | Conf | Size | Band | Spoken | Notes |
|---|---|---|---|---|---|---|
| 1 m | | | | | | |
| 2 m | | | | | | |
| 3 m | | | | | | |
| 5 m | | | | | | |
| 8 m | | | | | | |

## Result

- At what distance does each class first become "near" (Warning)? ______
- Is anything spoken that is clearly too far away? ______
- Is anything close left silent? ______
- Suggested `alert_gate.min_size` / `near_size` changes, with the sizes above: ______
