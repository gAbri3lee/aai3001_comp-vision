# Manual test sheet

These checks need a person, a real camera, speakers or a Roboflow account.
Automated tests cannot do them. Tick each box and add a note with the date.

Start the app with `python src/webapp.py --port 5001 --no-depth` and open
`http://127.0.0.1:5001` (add `/?debug=1` to see ignored boxes and sizes).

- [ ] Gabriel's latest push has been pulled and reviewed before any work starts.
- [ ] Live webcam works at the default camera index. (macOS may ask for Terminal
      camera permission.)
- [ ] Spoken alerts are audible, understandable and not repeated too often while walking.
- [ ] A person in the background does not trigger speech; the same person up close does.
- [ ] A person at the far side edge is ignored; a near one is spoken as "on your left"
      or "on your right".
- [ ] A pillar is sometimes missed. This is expected (weak class), not a bug.
- [ ] A chair or table in the path gives "Object detected ahead"; one at the side
      or far away is silent.
- [ ] Banner text, highlighted (magenta) box and spoken phrase match.
- [ ] Local and cloud give similar behaviour on the same walk. Note the cloud delay.
- [ ] Tape-measure distance test completed and results saved in
      `docs/DISTANCE_TEST_SHEET.md`.
- [ ] Gate sizes reviewed against the distance test and the Phase D proposal in
      `docs/GUI_TEST_REPORT.md`; `alert_gate.calibrated` set to true only after that.
- [ ] Cloud endpoint tested from a second device (needs a real hosted URL, not localhost).
- [ ] The old Roboflow key has been revoked in Roboflow and a new private key
      generated, then stored only as the `ROBOFLOW_API_KEY` environment variable or
      a Kaggle secret. This cannot be done from code.

## Interface (second hand-over)

Start the cloud service (`curl http://127.0.0.1:9100/health` should say ready),
then the app with both detectors and speech on:
`python src/webapp.py --port 5001 --no-depth --endpoint http://127.0.0.1:9100`.

Switch and fallback:
- [ ] The badge reads `LOCAL, YOLOv8n` at start; Cloud switches it to
      `CLOUD, RF-DETR Nano` and shows "Cloud mode: camera frames are sent to the server."
- [ ] Without `--endpoint` the Cloud button reads "Cloud unavailable" and cannot be pressed.
- [ ] Switching Local to Cloud mid-walk does not confuse the speech: the next
      hazard is spoken once, from the new detector, with no burst of old phrases.
- [ ] Stopping the cloud container (`docker stop visionaid-cloud`) shows "Cloud
      unreachable, using local." within about 3 seconds and alerts continue on local.
- [ ] Retry cloud stays on local while the container is stopped, and switches back
      after `docker start visionaid-cloud` once `/health` is ready.

Pause, mute and speech:
- [ ] Spoken alerts are audible; Mute stops the current phrase at once.
- [ ] While muted the banner and lists still update; Unmute brings speech back.
- [ ] Pause freezes the video with a "Paused" label and stops speech; Resume restarts.
- [ ] Test voice says "VisionAid voice test"; Replay last alert repeats the last alert.
- [ ] The rate slider changes the speed of the next phrase (try 120 and 260).
- [ ] The spoken log lists what was actually heard, newest first; Clear empties it.

Upload and samples:
- [ ] Uploading a phone photo (JPEG) shows the annotated image, banner and list,
      and Back to live returns to the camera.
- [ ] A PNG works; a file over 10 MB or a non-image gives a clear message.
- [ ] Samples 1 to 20 run; Speak result speaks only when ticked and not muted.

Error messages:
- [ ] Unplugging the camera (or closing the laptop camera cover) shows "Camera not
      found. Check the connection." or "No video. Waiting for camera.", the page
      stays up, and it recovers when plugged back in.
- [ ] With `say` broken or audio unavailable the page says "Speech is unavailable.
      Alerts show on screen only." (optional; hard to provoke).

Keyboard and layout:
- [ ] Space pauses, M mutes, C switches; none of them fire while the rate slider,
      the sample list or the file chooser has focus.
- [ ] Tab reaches every control with a visible yellow focus outline.
- [ ] On the projector (1920x1080) the banner is readable from the back of the room.
- [ ] On the small laptop (1366x768) the video, banner, legend, header controls and
      status fit without scrolling the page.
- [ ] A marker can understand the priority from the legend and banner without
      explanation (ask someone who has not seen the app).
- [ ] With a screen reader (macOS VoiceOver, Cmd+F5) a new banner alert is announced.

## Upload demo (third hand-over)

Start the app as above. Put a few test-split images in `demo/<class>/` and a
video in `demo/videos/` (see `demo/README.md` and `docs/DEMO_VIDEO_GUIDE.md`).

- [ ] Upload a real phone video (landscape, MP4, 30 to 60 seconds) and play it
      start to finish on the demo laptop.
- [ ] Speech during video playback is audible, in time with the banner, and Mute
      stops it at once.
- [ ] Every class that appears in the video is detected at least once. Note any
      misses, expecting Pillar to be weakest.
- [ ] A portrait phone video is oriented correctly and Recognition view behaves
      sensibly.
- [ ] Switching Local to Cloud during playback does not confuse the speech.
- [ ] Image batch of 10 test-split images per class gives sensible recognised
      counts. Record them for the report:

      | Class | Local recognised | Cloud recognised |
      |---|---|---|
      | Escalator | of 10 | of 10 |
      | People | of 10 | of 10 |
      | Pillar | of 10 | of 10 |
      | Stair | of 10 | of 10 |
      | Wet Floor Sign | of 10 | of 10 |

- [ ] Upload section is readable on the projector from a distance.
- [ ] Drag and drop and the file picker both work in the browser used for the demo.
- [ ] The end-of-video summary appears, Replay plays it again, and Back to camera
      returns to the live camera.

Notes:

| Date | Tester | Check | Result |
|---|---|---|---|
| | | | |
