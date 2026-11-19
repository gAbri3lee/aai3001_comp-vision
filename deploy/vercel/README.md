# VisionAid Photo Check: Vercel deployment

Upload-a-photo version of VisionAid. A static page (`public/index.html`) shrinks the photo in
the browser and POSTs it to a small Flask app (`app.py`, served as a Vercel Python function) that runs the YOLOv8n ONNX
export with only numpy, Pillow and onnxruntime (no torch), then draws the boxes.

This folder is its own Vercel project: **deploy from here, not from the repo root.** The root
`requirements.txt` pulls in torch and would exceed Vercel's function size limit. Vercel detects it as a
Python framework project, which needs a single web-app entrypoint, hence `app.py` (Flask).

## Deploy without GitHub

```sh
python deploy/vercel/prepare.py        # copies the ONNX model + config subset next to app.py (exports ONNX if needed)
npm i -g vercel
cd deploy/vercel
vercel login
vercel                                  # preview URL; answer the setup questions (no framework, no build command)
vercel --prod                           # production URL
```

## Check it

```sh
python deploy/vercel/app.py                             # local, no Vercel CLI: http://127.0.0.1:3000
python deploy/vercel/check_deployment.py http://127.0.0.1:3000
python deploy/vercel/check_deployment.py https://<your-app>.vercel.app   # golden parity + round-trip latency on the live site
```

`check_deployment.py` sends the 20 golden images and compares to `golden/expected/local_yolov8n_four_class.json`
(PASS = same classes/boxes). It also prints first/median/p95 round-trip latency for the report;
the first request includes the cold start.

## API

- `POST /api/detect` with a JPEG body (`Content-Type: image/jpeg`, up to ~4.4 MB, optional `?confidence=0..1`):
  shared output format (`detections`, `latency_ms`, `model`, plus `image_size`).
- `GET /api/health`: readiness, model name, class order.

## Notes

- ONNX export is fixed-shape 640x640 (letterboxed square). Results match Ultralytics running the same
  ONNX file exactly, but can differ slightly from the `.pt` model on non-square photos, which the `.pt` pipeline pads to a
  minimum rectangle instead.
- No alert gate: single photos have no distance/cooldown context. The message is simply the most urgent class
  (Stair > Escalator > Wet Floor Sign > People) and where its box sits (left / ahead / right).
- Uploaded photos are processed in memory and not stored or logged (`log_message` is disabled).
- The endpoint is public. Add Vercel Deployment Protection or an access check before sharing widely.
