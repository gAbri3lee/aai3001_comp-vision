# VisionAid Video demo guide on GUI (Local and Cloud)

The local GUI processes videos through the same detection and alert rules it
uses for the webcam. Use the bundled clips to check playback, or record a
walking clip to show how alerts change as objects move into view.

## Play a sample video clip from the Demo library

1. Open **Upload → Demo library** in the local GUI and click **Reload**.
2. Under **Videos**, choose a field sequence or the VivoCity atrium clip and click **Play**.
3. In **Alert** view, check the banner and listen to the alert
   with speech enabled and the GUI unmuted. Switch to **Recognition** to inspect the model's detections.
4. When **Video finished** appears, review the hazard type. Click **Replay**
   to run the clip again or **Back to camera** to return to the live view.

See the [demo library](../demo/README.md) for the clip sources and playback timing.

## Camera settings

- Hold the phone in **landscape** view (sideways). The alert rules were set for a
  landscape webcam. In a portrait video things look taller, so the alerts behave
  differently.
- Record in 1080p or 720p. Use H.264 MP4 if available for playback on the demo
  laptop. The GUI accepts MP4, MOV, and WebM.
- Keep it **30 to 60 seconds** long. The app accepts at most 120 seconds and
  150 MB.
- Hold the phone steady at chest height, pointing straight ahead, as a body
  camera would. Walk slowly.
- Do not zoom. Use good lighting and avoid fast turns, which blur the picture.

## What to film

The four trained classes in the live GUI are **Escalator, People, Stair, and Wet Floor Sign**.

1. Film one class at a time. Start a few metres back, move closer, and keep
   the subject in view for a few seconds before moving on.
2. Include at least one pass with a person standing in the background or at the
   side, then have the person fill more of the frame near the centre. Compare
   the detections shown in **Recognition** with the announcements in **Alert**.
3. Film an actual **Wet Floor Sign**. The detection refers to the sign itself.

## Keep it honest

- Film new places. Do not reuse places or footage from the training set, or the
  model will look better than it really is.
- If you use photos instead, take new ones or use images from the test split.
- The end summary shows what the model recognised in that clip. For model
  accuracy on the labelled test splits, use the [README benchmarks](../README.md#detection-benchmarks-by-model-and-class).

## Upload your own Video to the GUI

- Use **Upload → Video → Choose video** or drag the file into the video drop
  zone. To keep it in the library, copy it to `demo/videos/` and click **Reload**
  under **Demo library**.
- Play the entire clip before the demo with speech enabled. Check the timing of
  the announcements and the end summary. If RF-DETR is connected, switch between
  **Local** and **Cloud** and replay the same clip for each detector.
