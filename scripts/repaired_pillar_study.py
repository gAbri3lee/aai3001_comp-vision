"""Matched original/repaired-label study, frozen backbone and BN statistics.

Train-only review, unchanged validation, no test input or automatic deployment.
"""
import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

from controlled_pillar_study import acceptance, FINGERPRINT
from pillar_gpu_pilot import evaluate, NAMES


def schedule_draws(images, scenes, positive, excluded, total=8247, seed=0):
    rng = random.Random(seed)
    groups, negatives = defaultdict(list), []
    for name in sorted(images):
        if name in excluded:
            continue
        if name in positive:
            groups[scenes[Path(name).stem]].append(name)
        else:
            negatives.append(name)
    queues = []
    for scene in sorted(groups):
        values = groups[scene] * 4
        rng.shuffle(values)
        queues.append(deque(values[:32]))
    target = min(round(total * .2), sum(map(len, queues)))
    chosen = []
    while len(chosen) < target:
        for queue in queues:
            if queue and len(chosen) < target:
                chosen.append(queue.popleft())
    rng.shuffle(negatives)
    if len(negatives) < total - target:
        raise ValueError('Insufficient unique negatives for the declared training budget')
    chosen += negatives[:total-target]
    rng.shuffle(chosen)
    return chosen


def main():
    import torch
    import yaml
    from ultralytics import YOLO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['control', 'repaired'], required=True)
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    roots = list(Path('/kaggle/input').rglob('pilot_input.json'))
    if len(roots) != 1:
        raise ValueError('Attach exactly the approved private input')
    source = roots[0].parent
    if (source/'test').exists():
        raise ValueError('Test input prohibited')
    manifest = (source/'split_manifest.csv').read_text()
    if hashlib.md5(manifest.encode()).hexdigest() != FINGERPRINT:
        raise ValueError('Unexpected split')
    rows = list(csv.DictReader(manifest.splitlines()))
    splits = defaultdict(set)
    for row in rows:
        splits[row['scene']].add(row['split'])
    if any(len(s) > 1 for s in splits.values()):
        raise ValueError('Recorded scene leakage')
    patch = Path('configs/pillar_label_repair_2.json')
    decisions = json.loads(patch.read_text())
    root = Path('/tmp/visionaid_repaired')
    subprocess.run([sys.executable, 'scripts/apply_pillar_repairs.py', '--dataset',
                    str(source), '--decisions', str(patch), '--output', str(root)], check=True)
    if len(list((root/'valid/images').iterdir())) != 925:
        raise ValueError('Unexpected validation membership')
    train = sorted((root/'train/images').iterdir())
    if len(train) != 8247:
        raise ValueError('Unexpected training membership')
    positive = {p.name for p in train if any(line.split()[0]=='2'
        for line in (root/'train/labels'/(p.stem+'.txt')).read_text().splitlines() if line.strip())}
    scenes = {r['stem']:r['scene'] for r in rows if r['split']=='train'}
    draws = schedule_draws([p.name for p in train], scenes, positive,
                           set(decisions['excluded_train_images']))
    if args.arm == 'control':
        # Same pixel inputs/draws, only reviewed labels differ between arms.
        for row in decisions['records']:
            name = Path(row['image']).stem+'.txt'
            shutil.copyfile(source/'train/labels'/name, root/'train/labels'/name)
    weights = root/'yolov8n_v4.pt'
    metadata = json.loads(roots[0].read_text())
    if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata['weights_sha256']:
        raise ValueError('Original checkpoint mismatch')
    out = Path('/kaggle/working')/('repair_'+args.arm)
    out.mkdir(exist_ok=False)
    (out/'draw_names.txt').write_text('\n'.join(draws)+'\n')
    listing = out/'draws.txt'
    listing.write_text(''.join(str(root/'train/images'/name)+'\n' for name in draws))
    config = out/'data.yaml'
    config.write_text(yaml.safe_dump(dict(path=str(root),train=str(listing),val='valid/images',nc=5,names=NAMES)))
    plan = dict(arm=args.arm,epochs=args.epochs,seed=0,freeze=10,bn_running_statistics_frozen=True,
        batch=16,imgsz=640,lr0=.0001,training_draws=len(draws),unique_images=len(set(draws)),
        pillar_draws=sum(n in positive for n in draws),max_pillar_draws_per_scene=32,
        max_pillar_image_repeats=4,excluded_images=len(decisions['excluded_train_images']),
        draw_names_sha256=hashlib.sha256((out/'draw_names.txt').read_bytes()).hexdigest(),
        decisions_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
        validation_changed=False,test_used=False,application_changed=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    print('REPAIR_PLAN',json.dumps(plan),flush=True)
    baseline = YOLO(str(weights))
    before = evaluate(baseline,root,out,'baseline')
    del baseline
    torch.cuda.empty_cache()
    model = YOLO(str(weights))
    def verify_loader(trainer):
        if len(trainer.train_loader.dataset) != len(draws):
            raise ValueError('Loader changed the matched budget')
    def freeze_bn(trainer):
        # This event follows trainer.model.train(), unlike epoch-start.
        if getattr(trainer,'_repair_bn_epoch',None) != trainer.epoch:
            count = 0
            for layer in trainer.model.modules():
                if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):
                    layer.eval()
                    count += 1
            trainer._repair_bn_epoch = trainer.epoch
            print('BN_FROZEN',trainer.epoch,count,flush=True)
    model.add_callback('on_train_start',verify_loader)
    model.add_callback('on_train_batch_start',freeze_bn)
    model.train(data=str(config),epochs=args.epochs,patience=0,batch=16,imgsz=640,
        freeze=10,device=0,seed=0,deterministic=True,workers=2,optimizer='AdamW',
        lr0=.0001,close_mosaic=0,mosaic=.25,scale=.2,translate=.05,
        warmup_epochs=1.,warmup_bias_lr=.0001,cos_lr=True,lrf=.1,
        project=str(out),name='candidate',save_period=1)
    del model
    torch.cuda.empty_cache()
    results=[]
    for checkpoint in sorted((out/'candidate/weights').glob('epoch*.pt')):
        candidate=YOLO(str(checkpoint))
        after=evaluate(candidate,root,out,'validation_'+checkpoint.stem)
        passed,regressions=acceptance(before,after)
        record=dict(checkpoint=checkpoint.name,metrics=after,gate_passed=passed,
                    other_class_regressions=regressions)
        results.append(record)
        print('REPAIR_CHECKPOINT',json.dumps(record),flush=True)
        del candidate
        torch.cuda.empty_cache()
    if len(results) != args.epochs:
        raise ValueError('Missing checkpoint evaluations')
    selected=max(results,key=lambda r:(r['gate_passed'],r['metrics']['per_class']['Pillar']['recall'],
        r['metrics']['per_class']['Pillar']['precision'],r['metrics']['map50']))
    shutil.copyfile(out/'candidate/weights'/selected['checkpoint'],out/'selected.pt')
    result=dict(plan=plan,before=before,selected=selected,checkpoints=results,
        selected_sha256=hashlib.sha256((out/'selected.pt').read_bytes()).hexdigest(),
        limitation='Exploratory one-seed partial AI label repair; human review and new field holdout remain necessary.')
    (out/'result.json').write_text(json.dumps(result,indent=2))
    shutil.make_archive('/kaggle/working/repair_'+args.arm+'_review','zip',out)
    print('REPAIR_COMPLETE',json.dumps(selected),flush=True)


if __name__ == '__main__':
    main()
