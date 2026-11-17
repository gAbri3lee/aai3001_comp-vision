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

## Videos are not committed

Videos stay on the demo laptop. `demo/videos/` and all video files are listed in
`.gitignore`. See `docs/DEMO_VIDEO_GUIDE.md` for how to film the walking video.
Small example images may be committed; keep them small.
