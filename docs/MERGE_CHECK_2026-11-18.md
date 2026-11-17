# Pre-merge review and publishing steps — 18 November 2026

The four-class local fixes can be proposed to `main`. A disposable clone was used to simulate the merge; the real working branch remains `improve/pillar`, with its changes uncommitted. Nothing was pushed and no real merge was performed.

## Branch review

Fetched remote tips:

| Branch | Tip | Result |
| --- | --- | --- |
| main | `033aa630b8a76ac3c2f0b0d3353160d02e75e032` | Same base as the current working branch; simulated candidate merge has zero Git conflicts |
| gabriel | `9e1fd6e` | Already included in main |
| integration/main-ready | `58a69f6` | Already included in main |
| Matthias_Claude | `c6733bc` | Already included in main |
| cloud-deployment | `9a4c74a64ec0ad07a709c427675cce0a4900470d` | Additional photo service; simulated combined merge has zero Git conflicts, but separate model/deployment integration remains necessary |

The cloud branch's Vercel photo endpoint returns `model: yolov8n-onnx`, currently with five native classes including Pillar. The desktop Cloud adapter explicitly expects `rfdetr-nano`. These are different deployments; the new photo service cannot be treated as the desktop app's RF-DETR endpoint. The teammate should align the photo service's checkpoint/config/UI with the four-class scope and verify the intended hosted service separately. The local-fixes PR below does not include the cloud branch.

## Verification

- Final local full suite: **212 tests and 12 subtests passed** after correcting a reproduced timing race in a video test. The assertion now waits for the uploaded source and its results in the same status snapshot, rather than potentially reading a retiring camera frame. Final logs are under `artifacts/premerge-2026-11-18/`.
- Disposable combined checkout: 212 tests and 12 subtests passed, with two optional historical research-asset checks skipped because ignored datasets are not present in a clean clone. Those asset checks are included in the full local suite. Both Vercel ONNX detector tests also passed separately using the cloud branch's actual bundled model.
- Actual active four-class model: all **20 golden images passed**, with 20 matching detections and zero extras. Fresh HTTP checks passed for **11 real photo uploads and two real videos**, class IDs, corrupt-input isolation, replay/back-to-camera and temporary-file cleanup. Camera acquisition was replaced by an idle source; no speech was played.
- Python source compilation and `git diff --check` passed. Credential-marker/large-file screening of the candidate found no flagged items; that screening is not a comprehensive security audit.
- Earlier real Chrome workflows remain documented in `FOUR_CLASS_DEVELOPMENT_2026-11-18.md`. Physical distance, human audibility/usability and live hosted cloud remain outside this merge approval evidence.

The final local test and actual photo/video HTTP reports are recorded in `docs/evidence/premerge-2026-11-18.json` after completion. Automated passing checks support this specific revision; they do not establish that all possible inputs are free of bugs. If teammates change remote main, repeat the merge check before merging.

## Publish your checked branch

Open PowerShell in the repository and run each command in order. Stop if any command reports an error.

```powershell
cd "C:\Users\Asus\Desktop\Computer Vision and DL\Dataset\visionaid-latest"
git switch improve/pillar
git fetch origin
git add -A
git diff --cached --check
git diff --cached --stat
git commit -m "Fix runtime and upload reliability; align active four-class deployment"
git push -u origin improve/pillar
```

`git diff --cached --stat` lets you review the files before committing. The checked change set includes the earlier four-class migration, its required helpers/tests and research documentation. Ignored data, model checkpoints, calibration and local authentication files are excluded. The first push sets this branch's upstream to `origin/improve/pillar`; it currently tracks `origin/main`.

On GitHub, open **Pull requests → New pull request** in `gAbri3lee/aai3001_comp-vision`. Set **base: main** and **compare: improve/pillar**. Check **Files changed**, confirm GitHub reports no conflicts, then choose **Create pull request → Merge pull request → Confirm merge**. If branch protection requires review or checks, satisfy those before merging. Use a normal merge and keep the branch available until your team has verified the demo.

After the PR is merged, your teammates can pull main. Keep your local working directory clean before switching branches or pulling. Do not force-push or use a hard reset to resolve unexpected differences.

## Model/data handoff required

The active four-class checkpoint is ignored by Git. A code-only pull does not supply it. Extract `artifacts/four-class-handoff/visionaid-four-class-model-config.zip` into the teammate's repository root after pulling the matching code. Confirm the checkpoint at `models/local/yolov8n_v4_four_class.pt` has SHA256:

```text
58626a2662586f1a9c55de6a8b5aa3843dbbcb29f9e8ddc4455fd0583ceb9a39
```

The dataset ZIP is a separate training handoff; it is not needed for ordinary app startup. Preserve source attribution. Do not force-add full datasets, model weights or Kaggle credentials to this PR. Silent local launch is `run_demo.bat --no-tts`.
