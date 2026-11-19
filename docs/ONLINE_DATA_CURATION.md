# Online Singapore mall data

User requested online sourcing on 17 November 2026 because original team footage is unavailable. This is online curation, not original team photography. The rubric permits curated existing data, but completion still requires real annotation and review, not copying annotations or simply downloading images.

configs/online_source_candidates.json fixes membership and venue roles before model predictions. scripts/fetch_online_sources.py retrieves Commons author, source page, licence, source hash and downloaded SHA256. The first acquisition has 12 still images and one real 9.5-second VivoCity video, across eight Singapore venues. National Gallery is explicitly supplemental non-mall imagery. VivoCity and Nex are reserved external challenges; Marina Bay Sands, Plaza Singapura, Suntec City, Capitol Piazza and Resorts World Sentosa are development candidates. These are acquisition roles within the new collection, not proof that v4/pretraining never saw those venues.

Full metadata, downloaded assets and ATTRIBUTION.md are under data/online_sg_malls. Do not put these files into training automatically. Preserve attribution and each file's share-alike conditions when distributing adaptations. Resized previews and video conversion are modifications and must be declared. AI-created target boxes require team review and must not be called team-approved labels.

## Annotation and release rules

1. Audit exact/perceptual duplicates against all v4 splits and among new sources. Also inspect sources/venues manually; hash separation alone cannot prove independent scenes or pretraining independence.
2. Freeze a clear Pillar scope. The v4 class is broad and mixes columns, posts, trees and fixtures. Record the intended target before removing or relabelling these categories. Keep class IDs fixed.
3. Annotate all five classes visible in each training image, including occluded/truncated objects under an explicit policy. A box around floor is not Pillar. Do not turn unresolved examples into blank-label backgrounds. Record minimum supported size and ignore regions if annotation cannot be complete.
4. Have a second reviewer check each image; save original and corrected labels, reviewer, reason, source/scene ID and hashes. Reject uncertain images from training until resolved.
5. Keep related photographs/crops/video frames in one source group. Keep the frozen external challenge entirely out of training and validation selection.

annotations/online_primary_targets.json records three selected large-column targets drawn after full-resolution visual inspection and before predictions. They are deliberately incomplete image annotations. They can support primary-target recognition checks, but must never be used as exhaustive five-class YOLO training/evaluation labels. No all-class mAP or false-positive metric can be claimed from them.

## What this collection cannot establish

Thirteen files are a seed collection and challenge demonstration, not enough evidence to resolve Pillar generally. Architectural photos, close-up column signs and an old escalator video differ from head-height walking footage. Wet Floor Sign coverage remains missing in this initial licensed Singapore collection. Additional independent pillar scenes, clutter/fixture negatives, signs, low light, motion blur and occlusion are still needed. Never invent physical distances from these images or claim five-class field accuracy from a selected demonstration.

## Executed initial challenge

All 12 stills decoded and ran through the actual trained local detector via POST /analyze. The real Commons VivoCity Theora video was converted to MP4 without audio, retaining dimensions and source fps (284 frames at 30 fps). POST /source playback ended; 160 frames were processed under latest-frame scheduling, replay was accepted and temporary copies were cleaned up. This was Flask application-route testing with real model outputs and only camera acquisition replaced; the browser was not operated in this pass. General detection, speech, depth and cloud were disabled.

The selected large structural columns SG001/SG002/SG013 were missed. No trained class was reported during the 160 processed video frames. Some other stills produced People or Stair detections. Suntec SG009 produced Pillar boxes on advertising/shop regions rather than the desired structural columns, confirmed by full-image inspection. This is a substantial external-domain limitation; it must not be hidden behind the successful software tests. The sample is selected and incompletely annotated, so it supplies failure examples rather than five-class accuracy or a measured range.

The duplicate screen compared all 12 stills against all 10,092 v4 images. No exact SHA256 duplicate was found. Two low-distance dHash matches (SG012 and SG013) were visually checked and are different content: the v4 matches show a vending-machine/wet-floor-sign view and a downward pavement/shoe view. These are hash collisions, not confirmed source duplicates. The screen does not prove venue, crop or pretraining independence. See docs/evidence/online-overlap.json.
