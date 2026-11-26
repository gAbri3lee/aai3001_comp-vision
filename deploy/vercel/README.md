# VisionAid Photo and Video Check: Vercel deployment

Upload-a-photo-or-video version of VisionAid. A static page (`public/index.html`) shrinks the photo in
the browser and POSTs it to a small Flask app (`app.py`, served as a Vercel Python function) that runs the YOLOv8n ONNX
export with only numpy, Pillow and onnxruntime (no torch), then draws the boxes. A video is played in the
browser and checked frame by frame through the same endpoint (see "Video" below).

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

## Video

1. Open the site, click **Choose a photo or video** (or drop a file on the page) and pick an MP4 or WebM file.
   Which formats play depends on the browser; MP4 (H.264) is the safest choice, and a file the browser cannot play
   shows a message instead of a player. (Tested so far in Microsoft Edge only.)
2. Press **Play**. Coloured boxes with the class name and confidence appear over the picture, and the alert line,
   detection list and timings underneath follow the most recent result. **Pause** stops playback and checking,
   **Restart** goes back to the start and plays, and choosing another file (photo or video) replaces the current one.
3. **Checks per second** (1, 2 or 3; default 2) sets how often a frame is sent.

How it works: the video never leaves the browser. About twice a second the page copies the current frame to a canvas,
shrinks it to at most 640 px on its longest side (the model's input size), encodes it as a JPEG (quality 0.8, roughly 10 to 50 KB)
and sends it to `POST /api/detect`. Nothing else is uploaded, and the server code and model are unchanged.

- **One request at a time.** A tick that fires while the previous request is still running is skipped and counted
  (**Skipped while busy** in the timings line), so a slow or cold function never builds a queue, and playback never waits for it.
- **Stale results are dropped.** Replies that arrive after Restart, a seek, or loading another file are discarded, and boxes
  disappear if no fresh result arrives for 2.5 s. Because a reply takes a round trip, boxes trail the picture slightly (typically a few tenths of a second), most visibly when the camera moves quickly.
- **Failures don't stop the video.** If the function is unreachable, slow (20 s timeout) or returns an error, the page shows the
  message, keeps playing, and retries with a growing delay (2, 4, 8 s, then every 8 s) until it recovers.
- **Alignment.** Boxes are scaled from the sent frame to the picture's on-screen rectangle (including black bars for portrait or
  ultra-wide clips) and redrawn when the player is resized.
- The video plays muted. Videos are not read aloud: the **Read the result aloud** option applies to photos only. Detection pauses while the tab is in the background.
- The deployed function is public, so a visitor who leaves a video playing sends about two requests per second. Use Deployment
  Protection if that matters, and raise or lower the rate in `public/index.html` (`<select id="rate">`) as needed.

## Check it

```sh
python deploy/vercel/app.py                             # local, no Vercel CLI: http://127.0.0.1:3000
python deploy/vercel/check_deployment.py http://127.0.0.1:3000
python deploy/vercel/check_deployment.py https://<your-app>.vercel.app   # golden parity + round-trip latency on the live site
```

```sh
pip install playwright                    # uses an installed Edge or Chrome; no browser download
python deploy/vercel/check_video_ui.py http://127.0.0.1:3000
python deploy/vercel/check_video_ui.py https://<your-app>.vercel.app [--bypass-secret <Deployment Protection bypass secret>]
```

`check_video_ui.py` drives a real browser through the video workflow (upload, Play, Pause, Resume, Restart, end of video,
replacing the file, resizing, slow and failing detectors, bad files) and checks the requests the page makes. Its slow and failing
cases are answered by the test itself; the rest use the real endpoint, so run it against the deployed URL to verify cloud video
detection. `pytest tests/test_vercel_video.py` covers the page's upload contract and the endpoint's handling of video frames.

`check_deployment.py` sends the 20 golden images and compares to `golden/expected/local_yolov8n_four_class.json`
(PASS = same classes/boxes). It also prints first/median/p95 round-trip latency for the report;
the first request includes the cold start.

## API

- `POST /api/detect` with a JPEG body (photos, and single video frames from the page) (`Content-Type: image/jpeg`, up to ~4.4 MB, optional `?confidence=0..1`):
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
