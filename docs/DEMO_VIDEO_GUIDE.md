# Filming the demo video

This is for whoever films the walking video that VisionAid plays during the demo.
The app plays the video as if it were the camera, so the audience sees and hears
what a user would. A good recording takes about 15 minutes.

## Camera settings

- Hold the phone in **landscape** (sideways). The alert rules were set for a
  landscape webcam. In a portrait video things look taller, so the alerts behave
  differently.
- Record in 1080p or 720p, as MP4 (H.264). This is the normal setting on most
  phones. On an iPhone, set Settings > Camera > Formats to "Most Compatible" so
  the file is H.264, not HEVC.
- Keep it **30 to 60 seconds** long. The app accepts at most 120 seconds and
  150 MB.
- Hold the phone steady at chest height, pointing straight ahead, as a body
  camera would. Walk slowly.
- Do not zoom. Use good lighting and avoid fast turns, which blur the picture.

## What to film

The model knows five things: Escalator, People, Pillar, Stair and Wet Floor Sign.

1. Film them one at a time. Start about 4 to 5 metres away, walk towards it, and
   keep it in view for 3 to 4 seconds before moving on to the next one.
2. Include at least one pass with a person standing in the background or at the
   side. The audience should see that far and side hazards stay quiet, and that
   the same person is announced when close and ahead.
3. **Pillar:** film a clear, full-height pillar from 3 to 5 metres. Pillar is the
   model's weakest class (it finds about 17 percent of pillars locally and 29
   percent with the cloud model), so expect some misses. A missed pillar is a known
   limitation, not a fault in the demo.
4. **Wet Floor Sign:** a sign is a warning sign. The app does not detect a wet
   floor.

## Keep it honest

- Film new places. Do not reuse places or footage from the training set, or the
  model will look better than it really is.
- If you use photos instead, take new ones or use images from the test split.
- The summary at the end of the video is a check on that video only. The tested
  accuracy is mAP50 0.658 (local) and 0.739 (cloud) on 920 test images.

## Getting it onto the demo laptop

- Copy the file into `demo/videos/` on the demo laptop. It then appears under
  Upload, Demo library, with a Play button. You can also drag it onto the Video
  drop zone.
- Do not commit the video to git. Video files are ignored on purpose.
- Play it once before the demo, with speech on, to check the sound and the
  summary at the end.
