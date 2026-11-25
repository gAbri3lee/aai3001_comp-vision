# Demo library

The page lists what is in this folder under Upload, Demo library, so a presenter
can click a prepared image or video instead of using a file dialog on stage.

## What goes where

| Folder | Put here |
|---|---|
| `demo/Escalator/` | JPEG or PNG images that show an escalator |
| `demo/People/` | images that show people |
| `demo/Stair/` | images that show stairs |
| `demo/Wet Floor Sign/` | images that show a wet floor sign |
| `demo/videos/` | MP4, MOV or WebM videos (at most 150 MB and 120 seconds) |

The folder name is the class the images show. "Analyse" on one image, or
"Analyse all" for a class, uses that class as the expected class, so the page
says whether each image was recognised and gives a count such as
"Stair: 8 of 10 recognised". At most 20 images of a class are analysed at once,
and each image must be at most 10 MB.

Press Reload in the Demo library section after adding files. The app reads the
folder every time; files are never copied or changed, and only the names are
shown. Hidden files and links that lead outside this folder are ignored.

## Which images to use

For the presentation, open **Upload → Demo library** and click each named image. These four JPEGs are unchanged frames from the cleaned test split of the [team's v4 dataset](https://universe.roboflow.com/potatos-workspace-kequm/version2-nvt5l/dataset/4), distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Each row gives the expected output in Recognition view.

| Photo | Class | Visible action | Source frame in cleaned split |
|---|---|---|---|
| Escalator landing | Escalator | Check the direction before stepping on. | `v4_a990ec4f18c9e82896d23a61` |
| Pedestrians in a concourse | People | Give people room to pass. | `v4_36865e6411895ebeda97805d` |
| Staircase with handrails | Stair | Slow down and check the next step. | `v4_82ffc5d62c37a214064c8bb4` |
| Caution sign beside a column | Wet Floor Sign | Slow down and watch your footing. | `v4_902e9525e3a91699e698afc0` |

Both GUI backends identified the four photos and placed a box over the labelled target. For a useful counterexample, open **Upload → Images → Sample 1**. Local YOLO missed the labelled wet-floor sign there. Explain that a sign can be present even when the GUI reports “Nothing recognised.” The separate sample picker includes 20 verification frames and three general-obstacle photos. [Individual image and video results](../docs/evidence/demo-media-audit-2026-11-24.json)

## Bundled clips and obstacle examples

**Walk to the subway stairs** follows a pedestrian from the pavement into a subway entrance. The 13-second sequence plays at its original speed. [Walking Into a Metro Station](https://www.pexels.com/video/walking-into-a-metro-station-6082575/) · [Pexels licence](https://www.pexels.com/license/).

**Following a commuter down the escalator** follows a passenger from behind. Look for both People and Escalator from the opening seconds. [Original clip by Anastasia](https://www.pexels.com/video/man-walking-down-subway-escalator-9751571/) · [Pexels licence](https://www.pexels.com/license/).

**VivoCity - escalator atrium** is in `demo/challenges/`. The GUI displayed no trained-class action during a complete playback of this wide Singapore mall shot. Use it to demonstrate a miss by uploading it manually. Source: [Vivo City Escalators by ElHeineken](https://commons.wikimedia.org/wiki/File:Vivo_City_Escalators.ogv), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The clip was transcoded and audio removed.

Four other clips use different, chronologically ordered frames from single videos in the [team's v4 export](https://universe.roboflow.com/potatos-workspace-kequm/version2-nvt5l/dataset/4), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Playback uses six sampled images per second. Original timing and filming locations are unverified. These are selected demonstrations. Source scene IDs in the v4 scene manifest are:

| Clip | Source scene |
|---|---|
| Escalator | `761978818_826474_mp4` |
| Pedestrian | `764212632_214446_mp4` |
| Stairs and pedestrians | `VID_20260108_094455_mp4` |
| Wet floor sign | `11_mp4` |

The wet-floor clip retains the full 10-second sequence in portrait format. Removing the black border gave the sign more of the detector input. On 60 matching frames (six per second), local YOLO reported the sign on 22 frames, up from 15, and RF-DETR reported it on 50, up from 27. These are counts on this clip at the usual thresholds. In recorded-video Alert view, a grey box shows a recognised hazard that has not triggered a warning.

The sample-image picker includes three Singapore mall crops that exercise the general obstacle detector. [Photo credits](obstacles/Credits.md)

Additional personal videos remain excluded by `.gitignore`.

**Passengers descending the central staircase** uses seconds 50–62 of the station footage at the original speed. Look for People and Stair. [Original clip by 3 Vintage Lenses](https://www.pexels.com/video/people-are-walking-up-the-stairs-at-a-train-station-19319640/) · [Pexels licence](https://www.pexels.com/license/).

**People passing through a shared space** is the original Intel IoT DevKit pedestrian detection video with its filename changed. [Intel sample](https://github.com/intel-iot-devkit/sample-videos/blob/master/people-detection.mp4), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
