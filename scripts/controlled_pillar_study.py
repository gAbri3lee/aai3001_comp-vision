"""Matched GPU experiments: control, curated data, and curated scene sampling.

All arms start at the original checkpoint, use the same number of training
draws/updates, and evaluate the unchanged validation split. Test is excluded.
"""
import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import random
import time

from pillar_gpu_pilot import NAMES, evaluate

FINGERPRINT = '49601080aac1f838190341d331785578'
QUARANTINE = {'IMG_8603_MOV', 'amel-1_mp4'}


def acceptance(before, after):
    b, a = before['per_class']['Pillar'], after['per_class']['Pillar']
    regressions = [name for name in NAMES if name != 'Pillar' and
        after['per_class'][name]['recall'] < before['per_class'][name]['recall'] - .03]
    passed = (a['recall'] >= b['recall'] + .10 and
              a['precision'] >= max(.75, b['precision'] - .02) and
              after['map50'] >= before['map50'] - .01 and not regressions)
    return passed, regressions


def balanced_draws(groups, negatives, total, seed, fraction=.25, repeats=4):
    """Round-robin scene exposure, with a hard cap on repeats per pillar image."""
    rng = random.Random(seed)
    target = round(total * fraction)
    queues = []
    for scene in sorted(groups):
        values = list(groups[scene]) * repeats
        rng.shuffle(values)
        queues.append(deque(values))
    chosen = []
    while len(chosen) < target:
        progressed = False
        for queue in queues:
            if queue and len(chosen) < target:
                chosen.append(queue.popleft())
                progressed = True
        if not progressed:
            raise ValueError('Too few reviewed pillar examples for the requested repeat cap')
    negatives = list(negatives)
    rng.shuffle(negatives)
    if len(negatives) < total - target:
        raise ValueError('Too few non-pillar examples for the fixed training budget')
    chosen += negatives[:total - target]
    rng.shuffle(chosen)
    return chosen


def main():
    import torch
    import yaml
    from ultralytics import YOLO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['control', 'curated', 'curated-balanced'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--input', type=Path, help='Prepared private pilot input root; otherwise discover on Kaggle')
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no silent CPU fallback')
    roots = [args.input] if args.input else [p.parent for p in Path('/kaggle/input').rglob('pilot_input.json')]
    if len(roots) != 1:
        raise ValueError('Attach exactly the prepared v4 private pilot input')
    root = roots[0].resolve()
    metadata = json.loads((root / 'pilot_input.json').read_text())
    manifest = (root / 'split_manifest.csv').read_text()
    if hashlib.md5(manifest.encode()).hexdigest() != FINGERPRINT:
        raise ValueError('Unreviewed data split')
    if (root / 'test').exists():
        raise ValueError('Study input must exclude test images')
    rows = list(csv.DictReader(manifest.splitlines()))
    scene_splits = defaultdict(set)
    for row in rows:
        scene_splits[row['scene']].add(row['split'])
    if any(len(splits) > 1 for splits in scene_splits.values()):
        raise ValueError('Scene leakage')
    weights = root / 'yolov8n_v4.pt'
    if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata['weights_sha256']:
        raise ValueError('Unexpected baseline checkpoint')
    train = sorted((root / 'train/images').glob('*'))
    val = sorted((root / 'valid/images').glob('*'))
    if len(train) != 8247 or len(val) != 925:
        raise ValueError('Unexpected input membership')
    scenes = {r['stem']: r['scene'] for r in rows if r['split'] == 'train'}
    if set(scenes) != {p.stem for p in train}:
        raise ValueError('Train manifest mismatch')
    eligible = train if args.arm == 'control' else [p for p in train if scenes[p.stem] not in QUARANTINE]
    rng = random.Random(args.seed)
    if args.arm == 'curated-balanced':
        groups, negatives = defaultdict(list), []
        for p in eligible:
            label = root / 'train/labels' / (p.stem + '.txt')
            if any(line.split()[0] == '2' for line in label.read_text().splitlines() if line.strip()):
                groups[scenes[p.stem]].append(p)
            else:
                negatives.append(p)
        schedule = balanced_draws(groups, negatives, len(train), args.seed)
    else:
        schedule = eligible + rng.choices(eligible, k=len(train) - len(eligible))
        rng.shuffle(schedule)
    if len(schedule) != len(train):
        raise ValueError('Unequal training budget')
    out = args.output / f'study_{args.arm}_seed{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True, exist_ok=False)
    listing = out / 'training_draws.txt'
    listing.write_text(''.join(str(p) + '\n' for p in schedule))
    config = out / 'data.yaml'
    config.write_text(yaml.safe_dump(dict(path=str(root), train=str(listing),
                                          val='valid/images', nc=5, names=NAMES)))
    plan = dict(arm=args.arm, seed=args.seed, epochs=args.epochs, training_draws=len(schedule),
        unique_training_images=len(set(schedule)), original_training_images=len(train),
        quarantined_scenes=[] if args.arm == 'control' else sorted(QUARANTINE),
        updates_per_epoch=(len(schedule) + 15)//16, batch=16, optimizer='AdamW', lr0=.0001,
        patience=0, close_mosaic=0, test_used=False, validation_changed=False,
        mosaic=.25, scale=.2, translate=.05, warmup_epochs=1., warmup_bias_lr=.0001,
        cos_lr=True, lrf=.1, checkpoint_selection='fixed gate then Pillar recall, precision, overall AP',
        labels_changed=False, fraction_pillar=.25 if args.arm == 'curated-balanced' else None,
        max_repeats_pillar=4 if args.arm == 'curated-balanced' else None,
        checkpoint_sha256=metadata['weights_sha256'], split_manifest_md5=FINGERPRINT,
        training_draws_sha256=hashlib.sha256(listing.read_bytes()).hexdigest())
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    baseline = YOLO(str(weights))
    if list(baseline.names.values()) != NAMES:
        raise ValueError('Wrong class IDs')
    before = evaluate(baseline, root, out, 'before_validation')
    del baseline
    torch.cuda.empty_cache()
    model = YOLO(str(weights))
    epoch_times = []
    def epoch_start(trainer):
        trainer._study_started = time.monotonic()
    def epoch_end(trainer):
        epoch_times.append(time.monotonic()-trainer._study_started)
        print('STUDY_EPOCH',json.dumps(dict(arm=args.arm,epoch=trainer.epoch+1,
            seconds=epoch_times[-1],remaining_training_minutes=(args.epochs-trainer.epoch-1)*
            sum(epoch_times)/len(epoch_times)/60)),flush=True)
    def verify_budget(trainer):
        if len(trainer.train_loader.dataset) != len(schedule):
            raise ValueError('Loader dropped training entries; matched budget no longer holds')
    model.add_callback('on_train_start', verify_budget)
    model.add_callback('on_train_epoch_start',epoch_start)
    model.add_callback('on_train_epoch_end',epoch_end)
    model.train(data=str(config), epochs=args.epochs, patience=0, batch=16, imgsz=640,
        device=0, seed=args.seed, deterministic=True, workers=2, optimizer='AdamW',
        lr0=.0001, close_mosaic=0, mosaic=.25, scale=.2, translate=.05,
        warmup_epochs=1., warmup_bias_lr=.0001, cos_lr=True, lrf=.1,
        project=str(out), name='candidate', save_period=1)
    del model
    torch.cuda.empty_cache()
    candidates = []
    # Evaluate saved epochs at the real deployment threshold. Overall AP's
    # best.pt alone can hide an otherwise useful weak-class checkpoint.
    checkpoints = sorted((out/'candidate/weights').glob('epoch*.pt'))
    if not checkpoints:
        raise ValueError('Per-epoch checkpoints were not saved')
    for checkpoint in checkpoints:
        candidate = YOLO(str(checkpoint))
        metrics = evaluate(candidate,root,out,'validation_'+checkpoint.stem)
        passed, regressions = acceptance(before,metrics)
        candidates.append(dict(checkpoint=str(checkpoint.relative_to(out)),metrics=metrics,
                               gate_passed=passed,other_class_regressions=regressions))
        print('STUDY_CHECKPOINT',json.dumps(candidates[-1]),flush=True)
        del candidate
        torch.cuda.empty_cache()
    selected = max(candidates,key=lambda c:(c['gate_passed'],
        c['metrics']['per_class']['Pillar']['recall'],
        c['metrics']['per_class']['Pillar']['precision'],c['metrics']['map50']))
    after,passed,regressions = selected['metrics'],selected['gate_passed'],selected['other_class_regressions']
    result = dict(plan=plan, before=before, after=after, gate_passed=passed,
        other_class_regressions=regressions, application_changed=False,
        selected_checkpoint=selected['checkpoint'],checkpoints=candidates,epoch_seconds=epoch_times,
        limitation='One seed is exploratory. Compare all arms at matched seeds, then confirm on untouched field data.')
    (out / 'study_result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
