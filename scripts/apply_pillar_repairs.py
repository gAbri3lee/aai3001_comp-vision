"""Build a separate, hash-verified TRAIN variant; validation/test stay untouched.

Images and labels are independent copies. Refuses a pre-existing output directory.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from path_utils import native_path
from pillar_label_repair import repair_label
from config import CLASS_NAMES


def copy_validation_split(root, output, expected_stems):
    """Copy holdout bytes, including paths beyond Windows' legacy limit."""
    image_sources=[p for p in (root/'valid/images').iterdir() if native_path(p).is_file()]
    if {p.stem for p in image_sources} != set(expected_stems):
        raise ValueError('Source validation membership mismatch')
    sources=[]
    for kind in ('images','labels'):
        (output/'valid'/kind).mkdir(parents=True)
        sources.extend(p for p in (root/'valid'/kind).iterdir()
                       if native_path(p).is_file() and p.suffix!='.cache')
    def copy_one(path):
        shutil.copyfile(native_path(path),native_path(output/path.relative_to(root)))
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(copy_one,sources))
    copied={p.stem for p in (output/'valid/images').iterdir()}
    if copied != set(expected_stems):
        raise ValueError('Copied validation membership mismatch')
    return len(image_sources)


def main():
    from PIL import Image
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--decisions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.dataset.resolve()
    output=args.output.resolve()
    if output.exists() or output==root or root in output.parents:
        raise ValueError('Use a new output directory outside the source dataset')
    manifest=(root/'split_manifest.csv').read_bytes()
    if hashlib.md5(manifest).hexdigest()!='49601080aac1f838190341d331785578':
        raise ValueError('Unexpected source split')
    rows=list(csv.DictReader(manifest.decode().splitlines()))
    members={r['stem']:r for r in rows if r['split']=='train'}
    patch=json.loads(args.decisions.read_text())
    if patch['split_manifest_md5']!=hashlib.md5(manifest).hexdigest():
        raise ValueError('Review belongs to a different split')
    decisions={r['image']:r for r in patch['records']}
    if len(decisions)!=len(patch['records']):
        raise ValueError('Merge edits to the same image before applying')
    changed={}
    for name,decision in decisions.items():
        if Path(name).name!=name or Path(name).stem not in members:
            raise ValueError('Only existing training images may be edited')
        if members[Path(name).stem]['scene']!=decision['scene']:
            raise ValueError('Recorded scene mismatch')
        image=native_path(root/'train/images'/name)
        if hashlib.sha256(image.read_bytes()).hexdigest()!=decision['image_sha256']:
            raise ValueError('Reviewed image changed')
        label=native_path(root/'train/labels'/(image.stem+'.txt'))
        original=label.read_bytes() if label.exists() else b''
        with Image.open(image) as frame:
            width,height=frame.size
        edited,added=repair_label(original,decision,width,height)
        changed[image.stem]=(edited,dict(image=name,scene=decision['scene'],
            before_sha256=hashlib.sha256(original).hexdigest(),after_sha256=hashlib.sha256(edited).hexdigest(),
            added_lines=added,removed_lines=decision.get('remove_lines',[]),reviewer=decision['reviewer']))
    images,labels=output/'train/images',output/'train/labels'
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    source_images=sorted((root/'train/images').iterdir())
    if {p.stem for p in source_images}!=set(members):
        raise ValueError('Source training membership mismatch')
    def copy_training(image):
        source=native_path(image)
        destination=native_path(images/image.name)
        shutil.copyfile(source,destination)
        label=native_path(root/'train/labels'/(image.stem+'.txt'))
        data=changed[image.stem][0] if image.stem in changed else label.read_bytes() if label.exists() else b''
        native_path(labels/(image.stem+'.txt')).write_bytes(data)
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(copy_training,source_images))
    count=len(source_images)
    if count!=len(members):
        raise ValueError('Training membership mismatch')
    (output/'split_manifest.csv').write_bytes(manifest)
    if (root/'yolov8n_v4.pt').exists():
        shutil.copyfile(root/'yolov8n_v4.pt',output/'yolov8n_v4.pt')
    # Exact holdout copies permit local-disk caching on Kaggle. No validation
    # annotation is edited, and test images are deliberately not copied/read.
    valid_count=copy_validation_split(root,output,{r['stem'] for r in rows if r['split']=='valid'})
    import yaml
    (output/'data.yaml').write_text(yaml.safe_dump(dict(path=str(output),train='train/images',
        val='valid/images',nc=5,names=list(CLASS_NAMES))))
    report=dict(training_images=count,validation_images=valid_count,changed_images=len(changed),
        original_labels_changed=False,validation_changed=False,test_changed=False,
        validation_source=str(root/'valid/images'),
        decisions_sha256=hashlib.sha256(args.decisions.read_bytes()).hexdigest(),
        changes=[item[1] for item in changed.values()],
        note='AI-assisted reviewed subset; not a certification of every training annotation')
    (output/'repair_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
