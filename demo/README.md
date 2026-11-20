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

The active library has four classes: Escalator, People, Stair and Wet Floor Sign.
Prefer the cleaned **test split** or new held-out photos. Selected examples are demonstrations, not accuracy estimates. Historical five-class metrics are archived; see `docs/FOUR_CLASS_MIGRATION_2026-11-17.md` for current provenance and limits.

## Bundled clips and obstacle examples

**VivoCity - escalator atrium** is in `demo/challenges/`. The GUI displayed no trained-class action during a complete playback of this wide Singapore mall shot. Use it to demonstrate a miss by uploading it manually. Source: [Vivo City Escalators by ElHeineken](https://commons.wikimedia.org/wiki/File:Vivo_City_Escalators.ogv), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The clip was transcoded and audio removed.

Four other clips use different, chronologically ordered frames from single videos in the [team's v4 export](https://universe.roboflow.com/potatos-workspace-kequm/version2-nvt5l/dataset/4), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Playback uses six sampled images per second. Original timing and filming locations are unverified. These are selected demonstrations. Source scene IDs in the v4 scene manifest are:

| Clip | Source scene |
|---|---|
| Escalator | `761978818_826474_mp4` |
| Pedestrian | `764212632_214446_mp4` |
| Stairs and pedestrians | `VID_20260108_094455_mp4` |
| Wet floor sign | `11_mp4` |

The sample-image picker includes three Singapore mall crops that exercise the general obstacle detector. [Photo credits](obstacles/Credits.md)

Additional personal videos remain excluded by `.gitignore`.

**People crossing a walkway** shows pedestrians moving through an outdoor scene. The bundled excerpt uses the first 20 seconds of the [OpenCV pedestrian sample](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi), previously converted to MP4 for the project. Use Recognition view to see people away from the centre of the path.

**People passing through a shared space** is the original Intel IoT DevKit pedestrian detection video with its filename changed. [Intel sample](https://github.com/intel-iot-devkit/sample-videos/blob/master/people-detection.mp4), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
