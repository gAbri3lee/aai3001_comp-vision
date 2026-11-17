"""Pre-test extension: reuse the existing Pillar classifier row instead of resetting it."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    import torch
    from ultralytics import YOLO
    out=ROOT/'artifacts/mall-whole-scene-study'
    while len(json.loads((out/'status.json').read_text())['completed'])<6:time.sleep(2)
    status=json.loads((out/'status.json').read_text())
    if status['test_evaluated']:raise RuntimeError('Cannot extend a protocol after test evaluation')
    protocol=json.loads((out/'frozen_protocol.json').read_text())
    protocol['predeclared']['arms'].append('pillar_warmstart')
    protocol['pretest_addendum']='Third arm registered before test predictions: same standard whole-image inputs/updates, initialise from original Pillar row. Development-only diagnosis motivated the change.'
    (out/'frozen_protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(4)
    for seed in [0,1,2]:
        model=YOLO(str(ROOT/'artifacts/mall-pillar-warmstart/pillar-row-initial.pt'))
        model.train(data=str(out/'standard/dataset/data.yaml'),epochs=30,patience=0,batch=4,imgsz=416,
            freeze=10,device='cpu',workers=0,seed=seed,deterministic=True,optimizer='AdamW',lr0=.001,
            warmup_epochs=1,warmup_bias_lr=.001,cos_lr=True,lrf=.1,mosaic=1.,close_mosaic=5,
            scale=.4,translate=.1,plots=False,project=str(out/'pillar_warmstart'),name='seed'+str(seed),save_period=-1)
        last=out/'pillar_warmstart'/('seed'+str(seed))/'weights/last.pt'
        status['completed'].append(dict(arm='pillar_warmstart',seed=seed,last_sha256=hashlib.sha256(last.read_bytes()).hexdigest()))
        (out/'status.json').write_text(json.dumps(status,indent=2));print('WARMSTART_DONE',seed,flush=True)


if __name__=='__main__':main()
