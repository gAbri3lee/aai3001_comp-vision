"""Ten-epoch GPU pilot; training-only scene-capped pillar oversampling.

Run on Kaggle with the prepared private input bundle. No test images are included
or evaluated. A candidate is never promoted automatically to the application.
"""
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

NAMES = ['Escalator', 'People', 'Pillar', 'Stair', 'Wet Floor Sign']
EPOCHS = 10
SEED = 0


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / area if area else 0


def evaluate(model, root, output, tag):
    """Deployment-threshold greedy matches, plus low-threshold AP."""
    import yaml
    config = output / 'baseline.yaml'
    config.write_text(yaml.safe_dump(dict(path=str(root), train='train/images', val='valid/images',
                                         nc=5, names=NAMES)))
    metrics = model.val(data=str(config), split='val', imgsz=640, batch=16, conf=.001,
                        iou=.7, device=0, plots=False, project=str(output), name=tag+'_ap')
    tp, fp, fn = Counter(), Counter(), Counter()
    images = sorted((root / 'valid/images').glob('*'))
    # A Python image list uses LoadPilAndNumpy, which batches the entire list.
    # Directory input uses the file loader and respects the bounded batch size.
    for prediction in model.predict(source=str(root / 'valid/images'), batch=16, stream=True, imgsz=640,
                                    conf=.35, iou=.7, rect=False, device=0, verbose=False):
        image = Path(prediction.path)
        height, width = prediction.orig_shape
        truths = defaultdict(list)
        label = root / 'valid/labels' / (image.stem+'.txt')
        for line in label.read_text().splitlines():
            cid, x, y, w, h = map(float, line.split())
            truths[int(cid)].append(((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height))
        used = defaultdict(set)
        for x1,y1,x2,y2,confidence,cid in sorted(prediction.boxes.data.cpu().tolist(),key=lambda r:-r[4]):
            cid = int(cid)
            available = [j for j in range(len(truths[cid])) if j not in used[cid]]
            best = max(available,key=lambda j:iou((x1,y1,x2,y2),truths[cid][j]),default=None)
            if best is not None and iou((x1,y1,x2,y2),truths[cid][best]) >= .5:
                used[cid].add(best); tp[cid] += 1
            else:
                fp[cid] += 1
        for cid in range(5):
            fn[cid] += len(truths[cid])-len(used[cid])
    report = dict(images=len(images), map50=float(metrics.box.map50), map50_95=float(metrics.box.map),
                  confidence=.35, per_class={name:dict(tp=tp[c],fp=fp[c],fn=fn[c],
                  precision=tp[c]/max(tp[c]+fp[c],1), recall=tp[c]/max(tp[c]+fn[c],1)) for c,name in enumerate(NAMES)})
    (output/(tag+'.json')).write_text(json.dumps(report,indent=2))
    return report


def main():
    import torch
    import yaml
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required. Enable a Kaggle GPU; no silent CPU fallback.')
    inputs = Path('/kaggle/input')
    candidates = list(inputs.rglob('pilot_input.json'))
    if len(candidates) != 1:
        raise RuntimeError('Attach the single prepared VisionAid pillar pilot input bundle')
    root = candidates[0].parent
    metadata = json.loads(candidates[0].read_text())
    manifest_text = (root/'split_manifest.csv').read_text()
    if hashlib.md5(manifest_text.encode()).hexdigest() != '49601080aac1f838190341d331785578':
        raise RuntimeError('Wrong scene split')
    rows = list(csv.DictReader(manifest_text.splitlines()))
    scene_splits = defaultdict(set)
    for r in rows: scene_splits[r['scene']].add(r['split'])
    if any(len(s)>1 for s in scene_splits.values()): raise RuntimeError('Scene leakage')
    if (root/'test').exists(): raise RuntimeError('Pilot bundle must exclude test images')
    train = sorted((root/'train/images').glob('*'))
    valid = sorted((root/'valid/images').glob('*'))
    if len(train)!=8247 or len(valid)!=925: raise RuntimeError('Unexpected data counts')
    weights = root/'yolov8n_v4.pt'
    if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata['weights_sha256']:
        raise RuntimeError('Checkpoint integrity failed')
    output = Path('/kaggle/working') / ('pillar_pilot_' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(exist_ok=False)
    model = YOLO(str(weights))
    if list(model.names.values()) != NAMES: raise RuntimeError('Wrong model taxonomy')
    before = evaluate(model,root,output,'before_validation')
    scenes = {r['stem']:r['scene'] for r in rows if r['split']=='train'}
    pillar_groups = defaultdict(list)
    for p in train:
        label = root/'train/labels'/(p.stem+'.txt')
        if any(line.split()[0]=='2' for line in label.read_text().splitlines() if line.strip()):
            pillar_groups[scenes[p.stem]].append(p)
    rng = random.Random(SEED)
    extra = []
    for scene in sorted(pillar_groups):
        members = sorted(pillar_groups[scene]); rng.shuffle(members)
        extra.extend(members[:50]*2)
    if not extra: raise RuntimeError('No pillar training examples')
    train_list = output/'train_scene_balanced.txt'
    train_list.write_text(''.join(str(p)+'\n' for p in train+extra))
    dataset_yaml = output/'pilot.yaml'
    dataset_yaml.write_text(yaml.safe_dump(dict(path=str(root),train=str(train_list),
                            val='valid/images',nc=5,names=NAMES)))
    plan = dict(epochs=EPOCHS,seed=SEED,base_training_images=len(train),extra_pillar_draws=len(extra),
                extra_cap_per_scene=100,pillar_training_scenes=len(pillar_groups),
                labels_changed=False,test_used=False,method='scene-capped extra pillar exposure',
                limitations='Pilot does not isolate extra training epochs from resampling; a matched control is needed for causal claims.')
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    print('PILOT PLAN',json.dumps(plan),flush=True)
    epoch_times=[]
    def started(trainer): trainer._pilot_epoch_start=time.monotonic()
    def ended(trainer):
        epoch_times.append(time.monotonic()-trainer._pilot_epoch_start)
        remaining=(EPOCHS-trainer.epoch-1)*sum(epoch_times)/len(epoch_times)
        print(f'PILOT TIMING epoch={trainer.epoch+1} seconds={epoch_times[-1]:.1f} estimated_remaining_minutes={remaining/60:.1f}',flush=True)
    model.add_callback('on_train_epoch_start',started)
    model.add_callback('on_train_epoch_end',ended)
    model.train(data=str(dataset_yaml),epochs=EPOCHS,patience=3,batch=16,imgsz=640,
                device=0,seed=SEED,workers=2,optimizer='AdamW',lr0=.0005,
                project=str(output),name='candidate',save_period=1,exist_ok=False)
    candidate = YOLO(str(output/'candidate/weights/best.pt'))
    after = evaluate(candidate,root,output,'after_validation')
    pillar_before,pillar_after = before['per_class']['Pillar'],after['per_class']['Pillar']
    other_regressions = [name for name in NAMES if name!='Pillar' and
                         after['per_class'][name]['recall'] < before['per_class'][name]['recall']-.03]
    passed = (pillar_after['recall'] >= pillar_before['recall']+.10 and
              pillar_after['precision'] >= max(.60,pillar_before['precision']-.02) and
              after['map50'] >= before['map50']-.01 and not other_regressions)
    result=dict(candidate_passes_pilot_gate=passed,other_class_regressions=other_regressions,
                before=before,after=after,application_changed=False,epoch_seconds=epoch_times,
                next_step='Review validation examples and run matched control/new field test; never auto-replace deployed weights.')
    (output/'pilot_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
