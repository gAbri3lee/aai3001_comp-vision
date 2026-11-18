"""CPU-feasible, frozen-feature Pillar classifier experiment.

Only the Pillar output row of each final 1x1 classification convolution can
change. Backbone, box regression and the four other output rows remain frozen.
Training uses v4 TRAIN only, minus documented quarantine scenes. Validation
selects among four predeclared interpolation strengths; test is never read.
This is a limited candidate experiment, not a promised generalization repair.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from config import CLASS_NAMES
from path_utils import native_path, read_bgr

ALPHAS = (.1, .25, .5, 1.)
VERSION = 2


def sha(path):
    return hashlib.sha256(native_path(path).read_bytes()).hexdigest()


def members(root, negative_cap=4):
    manifest = root / 'split_manifest.csv'
    if hashlib.md5(manifest.read_bytes()).hexdigest() != '49601080aac1f838190341d331785578':
        raise ValueError('Unexpected dataset split')
    rows = list(csv.DictReader(manifest.read_text().splitlines()))
    quarantine = json.loads((ROOT / 'configs/v4_training_quarantine.json').read_text())
    excluded = {r['scene'] for r in quarantine['scenes']}
    train_rows = [r for r in rows if r['split'] == 'train' and r['scene'] not in excluded]
    train = {p.stem:p for p in (root / 'train/images').iterdir()}
    valid = sorted((root / 'valid/images').iterdir())
    if len(train) != 8247 or len(train_rows) != 8224 or len(valid) != 925:
        raise ValueError('Unexpected data membership')
    positive,negative = set(),defaultdict(list)
    for row in train_rows:
        path = native_path(root/'train/labels'/(row['stem']+'.txt'))
        labels = path.read_text() if path.exists() else ''
        if any(line.split()[0]=='2' for line in labels.splitlines() if line.strip()):
            positive.add(row['stem'])
        else:
            negative[row['scene']].append(row['stem'])
    selected = set(positive)
    rng = random.Random(0)
    for scene in sorted(negative):
        values = sorted(negative[scene])
        rng.shuffle(values)
        selected.update(values[:negative_cap])
    return [(train[r['stem']],r['scene']) for r in train_rows if r['stem'] in selected], valid


def select_training_anchors(boxes, truths, scores, sizes, seed):
    """Positive proposals overlap labelled pillars; ambiguous overlap is ignored.

    Cap positives per truth/feature level, and mine hard plus random negatives.
    This does not create new ground-truth boxes from model predictions.
    """
    import torch
    from ultralytics.utils.metrics import box_iou
    overlaps = box_iou(boxes, truths) if len(truths) else boxes.new_zeros((len(boxes),0))
    quality = overlaps.max(1).values if len(truths) else boxes.new_zeros(len(boxes))
    positives = []
    offset = 0
    for size in sizes:
        for k in range(len(truths)):
            values = overlaps[offset:offset+size,k]
            indices = torch.where(values >= .5)[0]
            if len(indices):
                indices = indices[values[indices].argsort(descending=True)[:8]]
                positives.extend((indices+offset).tolist())
        offset += size
    eligible = torch.where(quality <= .2)[0]
    hard = eligible[scores[eligible].argsort(descending=True)[:64]]
    generator = torch.Generator().manual_seed(seed)
    random = eligible[torch.randperm(len(eligible),generator=generator)[:32]]
    chosen = torch.tensor(sorted(set(positives + hard.tolist() + random.tolist())),dtype=torch.long)
    target = quality[chosen].clone()
    target[target <= .2] = 0
    return chosen, target, dict(ground_truth_pillars=len(truths),
        covered_truths=int((overlaps.max(0).values >= .5).sum()) if len(truths) else 0)


def cache(args, train, valid):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.ops import xywh2xyxy
    model = YOLO(str(args.model)).model.float().eval()
    if list(model.names.values()) != list(CLASS_NAMES):
        raise ValueError('Unexpected class IDs')
    head = model.model[-1]
    features = {}
    hooks = [branch[-1].register_forward_pre_hook(
        lambda module, inputs, level=i: features.update({level:inputs[0].detach()}))
        for i,branch in enumerate(head.cv3)]
    letterbox = LetterBox(new_shape=(640,640),auto=False)
    started = time.monotonic()
    try:
        for split, entries in [('train',train),('valid',[(p,'') for p in valid])]:
            folder = args.output / 'cache' / split
            folder.mkdir(parents=True,exist_ok=True)
            for index,(path,scene) in enumerate(entries):
                destination = folder / f'{index:05d}.npz'
                if destination.exists():
                    continue
                frame = read_bgr(path)
                if frame is None:
                    raise ValueError(f'Image decode failed: {path}')
                h,w = frame.shape[:2]
                image = letterbox(image=frame)
                tensor = torch.from_numpy(np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))).float()[None]/255
                with torch.inference_mode():
                    decoded,raw = model(tensor)
                    sizes = [features[i].shape[-2]*features[i].shape[-1] for i in range(3)]
                    level = torch.repeat_interleave(torch.arange(3),torch.tensor(sizes))
                    combined = torch.cat([features[i][0].flatten(1).T for i in range(3)])
                    logits = raw['scores'][0,2]
                    if combined.shape != (8400,80):
                        raise ValueError('Unsupported head feature layout')
                    data = dict(name=path.name,height=h,width=w)
                    if split == 'train':
                        label_path = native_path(args.dataset/'train/labels'/(path.stem+'.txt'))
                        labels = label_path.read_text() if label_path.exists() else ''
                        boxes = [list(map(float,row.split()[1:])) for row in labels.splitlines()
                                 if row.strip() and row.split()[0]=='2']
                        truth = torch.tensor(boxes).reshape(-1,4)
                        truth[:,[0,2]] *= w
                        truth[:,[1,3]] *= h
                        truth = xywh2xyxy(truth)
                        scale = min(640/h,640/w)
                        truth *= scale
                        truth[:,[0,2]] += round((640-w*scale)/2-.1)
                        truth[:,[1,3]] += round((640-h*scale)/2-.1)
                        chosen,target,coverage = select_training_anchors(
                            xywh2xyxy(decoded[0,:4].T),truth,logits,sizes,index)
                        data.update(features=combined[chosen].numpy().astype(np.float16),
                            logits=logits[chosen].numpy(),level=level[chosen].numpy().astype(np.uint8),
                            target=target.numpy(),scene=scene,**coverage)
                    else:
                        data.update(features=combined.numpy().astype(np.float16),
                            prediction=decoded[0].numpy(),logits=logits.numpy(),
                            level=level.numpy().astype(np.uint8))
                temporary = destination.with_suffix('.partial.npz')
                np.savez(temporary,**data)
                temporary.replace(destination)
                if (index+1)%100==0 or index==0:
                    print(f'CACHE {split} {index+1}/{len(entries)} elapsed={time.monotonic()-started:.1f}s',flush=True)
                if args.limit and index+1 >= args.limit:
                    break
    finally:
        for hook in hooks:
            hook.remove()


def train_head(args, entries):
    import numpy as np
    import torch
    import torch.nn.functional as F
    arrays, levels, logits, targets, scenes = [],[],[],[],[]
    covered,truth_count = 0,0
    for index,(path,scene) in enumerate(entries):
        with np.load(args.output/'cache/train'/f'{index:05d}.npz') as data:
            if str(data['name']) != path.name or str(data['scene']) != scene:
                raise ValueError('Training cache membership mismatch')
            covered += int(data['covered_truths'])
            truth_count += int(data['ground_truth_pillars'])
            arrays.append(data['features'].astype(np.float32))
            levels.append(data['level'].astype(np.int64))
            logits.append(data['logits'])
            targets.append(data['target'])
            scenes.extend([scene]*len(data['target']))
    x = torch.from_numpy(np.concatenate(arrays))
    x = torch.cat((x,torch.ones((len(x),1))),1)
    level = torch.from_numpy(np.concatenate(levels))
    original = torch.from_numpy(np.concatenate(logits))
    target = torch.from_numpy(np.concatenate(targets))
    positive = target > 0
    if not torch.isfinite(x).all() or not torch.isfinite(original).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite cached features or targets')
    if positive.sum() < 100:
        raise ValueError('Insufficient positive proposals')
    # Equalize positive scene contribution without treating adjacent frames as
    # independent sites. Negative mining still spans every eligible train image.
    counts = Counter(scene for scene,is_positive in zip(scenes,positive.tolist()) if is_positive)
    weight = torch.ones(len(x))
    for i,(scene,is_positive) in enumerate(zip(scenes,positive.tolist())):
        if is_positive:
            weight[i] = 1/counts[scene]
    factor = min(10.,float((~positive).sum()/positive.sum()))
    weight[positive] *= factor*positive.sum()/weight[positive].sum()
    weight /= weight.sum()
    groups = [(x[level==k],original[level==k],target[level==k],weight[level==k]) for k in range(3)]
    delta = torch.zeros((3,81),requires_grad=True)
    optimizer = torch.optim.LBFGS([delta],lr=1,max_iter=100,line_search_fn='strong_wolfe')
    steps = 0
    def closure():
        nonlocal steps
        optimizer.zero_grad()
        loss = .001*delta.square().sum()
        for k,(features,base,truth,importance) in enumerate(groups):
            adjusted = base + features@delta[k]
            loss = loss + (F.binary_cross_entropy_with_logits(adjusted,truth,reduction='none')*importance).sum()
        loss.backward()
        steps += 1
        if steps%10==0:
            print(f'HEAD OPTIMIZATION step={steps} loss={float(loss.detach()):.6f}',flush=True)
        return loss
    optimizer.step(closure)
    np.save(args.output/'delta.npy',delta.detach().numpy())
    report = dict(rows=len(x),positive_rows=int(positive.sum()),positive_scenes=len(counts),
        positive_weight=factor,regularization=.001,optimizer='LBFGS',steps=steps,
        alphas=ALPHAS,train_images=len(entries),test_used=False,trainable_parameters=243)
    report.update(ground_truth_pillars=truth_count,truths_with_overlapping_frozen_proposals=covered,
                  frozen_localization_recall_limit=covered/max(truth_count,1))
    (args.output/'training.json').write_text(json.dumps(report,indent=2))
    print('TRAINING',json.dumps(report),flush=True)


def evaluate(args,valid):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    from evaluation import load_ground_truth, score_detections
    delta = torch.from_numpy(np.load(args.output/'delta.npy'))
    gt,paths = load_ground_truth(args.dataset/'valid',CLASS_NAMES)
    reports = []
    for alpha in (0.,)+ALPHAS:
        outputs = {}
        for index,path in enumerate(valid):
            with np.load(args.output/'cache/valid'/f'{index:05d}.npz') as data:
                assert str(data['name']) == path.name
                pred = torch.from_numpy(data['prediction'].copy())
                if alpha:
                    features = torch.from_numpy(data['features'].astype(np.float32))
                    levels = torch.from_numpy(data['level'].astype(np.int64))
                    change = torch.empty(len(features))
                    for k in range(3):
                        mask = levels==k
                        change[mask] = features[mask]@delta[k,:80]+delta[k,80]
                    pred[6] = torch.sigmoid(torch.from_numpy(data['logits'])+alpha*change)
                boxes = non_max_suppression(pred[None],conf_thres=.001,iou_thres=.7,max_det=300,nc=5)[0]
                scale_boxes((640,640),boxes[:,:4],(int(data['height']),int(data['width'])))
                outputs[path.name] = {'detections':[dict(class_id=int(r[5]),confidence=float(r[4]),
                    box_xyxy=r[:4].tolist()) for r in boxes]}
        result = score_detections(gt,paths,outputs,CLASS_NAMES,.35,['Escalator','Stair','Wet Floor Sign'])
        result['alpha'] = alpha
        reports.append(result)
        print('VALIDATION',json.dumps(result),flush=True)
        tag = 'baseline' if not alpha else f'alpha_{alpha:g}'
        (args.output/f'{tag}.json').write_text(json.dumps(result,indent=2))
        if alpha:
            model = YOLO(str(args.model))
            original = {k:v.clone() for k,v in model.model.state_dict().items()}
            with torch.no_grad():
                for k,branch in enumerate(model.model.model[-1].cv3):
                    branch[-1].weight[2,:,0,0] += alpha*delta[k,:80]
                    branch[-1].bias[2] += alpha*delta[k,80]
            changed = []
            for name,value in model.model.state_dict().items():
                if not torch.equal(value,original[name]):
                    if name not in {f'model.22.cv3.{k}.2.{p}' for k in range(3) for p in ('weight','bias')}:
                        raise AssertionError(f'Unexpected changed tensor: {name}')
                    assert torch.equal(value[[0,1,3,4]],original[name][[0,1,3,4]])
                    changed.append(name)
            model.save(str(args.output/f'{tag}.pt'))
        else:
            if result['per_class']['Pillar']['recall'] != 47/161:
                raise ValueError('Cached baseline does not reproduce expected validation recall')
    baseline = reports[0]
    for result in reports[1:]:
        b,a = baseline['per_class']['Pillar'],result['per_class']['Pillar']
        result['gate_passed'] = (a['recall'] >= b['recall']+.10 and a['precision'] >= max(.75,b['precision']-.02)
            and result['map50'] >= baseline['map50']-.01 and all(
                result['per_class'][n]['recall'] >= baseline['per_class'][n]['recall']-.03
                for n in CLASS_NAMES if n!='Pillar'))
    (args.output/'comparison.json').write_text(json.dumps(reports,indent=2))


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['cache','train','evaluate'],required=True)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/v4_scene_split')
    parser.add_argument('--model',type=Path,default=ROOT/'models/local/yolov8n_v4.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/pillar-head-recovery-scene-capped')
    parser.add_argument('--negative-cap',type=int,default=4,help='Non-pillar images per training scene; all eligible pillar images retained')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--limit',type=int,default=0,help='Cache smoke test only; never a smaller evaluation')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.negative_cap < 1:
        parser.error('--negative-cap must be positive')
    train,valid = members(args.dataset,args.negative_cap)
    metadata = dict(version=VERSION,weights_sha256=sha(args.model),
        manifest_sha256=sha(args.dataset/'split_manifest.csv'),
        quarantine_sha256=sha(ROOT/'configs/v4_training_quarantine.json'),
        training_images=len(train),validation_images=len(valid),test_used=False,
        negative_cap=args.negative_cap,training_members_sha256=hashlib.sha256(
            '\n'.join(p.name for p,_ in train).encode()).hexdigest())
    metadata_path = args.output/'input.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError('Cache input differs; use a new output directory')
    metadata_path.write_text(json.dumps(metadata,indent=2))
    (args.output/'training_members.json').write_text(json.dumps(
        [dict(image=p.name,scene=scene) for p,scene in train],indent=2))
    if args.stage == 'cache':
        cache(args,train,valid)
    elif args.stage == 'train':
        train_head(args,train)
    else:
        evaluate(args,valid)


if __name__ == '__main__':
    main()
