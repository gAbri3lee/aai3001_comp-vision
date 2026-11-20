"""Build matching, CRC-checked data/model handoffs from explicit local inputs."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'artifacts/four-class-handoff'
DATA=ROOT/'data/v4_four_class_clean'


def package(target, entries):
    temporary=target.with_suffix('.building.zip')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for source,name in entries:
            if not source.is_file() or source.is_symlink():raise ValueError('Missing or linked input '+str(source))
            archive.write(source,name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:raise ValueError('Archive CRC failure')
        if len(archive.namelist())!=len(set(archive.namelist())):raise ValueError('Duplicate archive paths')
    temporary.replace(target)
    return dict(bytes=target.stat().st_size,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),crc_verified=True)


if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'visionaid_config.json').read_text())
    contract=json.loads((DEST/'CLASS_CONTRACT.json').read_text())
    assert contract['classes']==config['classes']
    contract['training_annotation_revision']=config['dataset']['training_annotation_revision']
    contract['dataset_version']=config['dataset']['version']
    contract['cloud_native_classes']=config['classes']
    contract['native_cloud_checkpoint']=config['cloud']['file']
    contract['model_trained_on_annotation_revision']=False
    (DEST/'CLASS_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    entries=[]
    for p in sorted(DATA.rglob('*')):
        relative=p.relative_to(DATA)
        if p.is_file() and (len(relative.parts)==1 and p.suffix in {'.json','.yaml','.csv','.txt'}
                           or len(relative.parts)==3 and relative.parts[0] in {'train','valid','test'}
                           and relative.parts[1] in {'images','labels'} and p.suffix.lower() in {'.jpg','.jpeg','.png','.txt'}):
            entries.append((p,'visionaid-four-class-dataset/'+relative.as_posix()))
    assert sum('/images/' in name for _,name in entries)==9377
    assert sum('/labels/' in name for _,name in entries)==9377
    entries.append((ROOT/'annotations/four_class_training_review.json','visionaid-four-class-dataset/TRAINING_ANNOTATION_REVIEW.json'))
    sums={}
    sums['visionaid-four-class-dataset.zip']=package(DEST/'visionaid-four-class-dataset.zip',entries)
    model_entries=[(ROOT/p,p) for p in [
        'models/local/yolov8n_v4_four_class.pt','models/local/yolov8n_v4_four_class.onnx',
        config['cloud']['file'],'visionaid_config.json','docs/evidence/four-class-demo-benchmark.json',
        'docs/FOUR_CLASS_MIGRATION_2026-11-17.md','docs/evidence/four-class-migration-2026-11-17.json',
        'docs/FOUR_CLASS_DEVELOPMENT_2026-11-18.md','docs/evidence/four-class-development-2026-11-18.json',
        'annotations/four_class_training_review.json',
        'artifacts/four-class-study/cpu-comparison.json','artifacts/four-class-study/browser/report.json']]
    model_entries.append((DEST/'CLASS_CONTRACT.json','CLASS_CONTRACT.json'))
    sums['visionaid-four-class-model-config.zip']=package(DEST/'visionaid-four-class-model-config.zip',model_entries)
    (DEST/'SHA256SUMS.json').write_text(json.dumps(sums,indent=2)+'\n')
    print(json.dumps(sums,indent=2))
