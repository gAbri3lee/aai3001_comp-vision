"""Create a separate native four-output YOLOv8 checkpoint without retraining.

Only final independent class-logit rows are selected. Backbone, box regression,
DFL and remaining class weights are unchanged. Pre-NMS raw parity is verified;
post-NMS comparison is separately required before deployment.
"""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
KEEP=[0,1,3,4]
NAMES=['Escalator','People','Stair','Wet Floor Sign']


def convert(source, destination):
    import torch
    import ultralytics
    from ultralytics import YOLO
    if destination.exists():raise ValueError('Refusing to overwrite four-class checkpoint')
    original=YOLO(str(source));model=copy.deepcopy(original.model).cpu().float().eval()
    expected=['Escalator','People','Pillar','Stair','Wet Floor Sign']
    if list(original.names.values())!=expected:raise ValueError('Wrong source class order')
    head=model.model[-1]
    if head.nc!=5 or head.end2end or not hasattr(head,'cv3'):raise ValueError('Expected five-class non-end-to-end YOLO Detect head')
    for branch in head.cv3:
        old=branch[-1]
        if not isinstance(old,torch.nn.Conv2d) or old.out_channels!=5:raise ValueError('Unexpected classifier shape')
        new=torch.nn.Conv2d(old.in_channels,4,old.kernel_size,old.stride,old.padding,old.dilation,
                            old.groups,old.bias is not None,old.padding_mode)
        with torch.no_grad():
            new.weight.copy_(old.weight[KEEP])
            if old.bias is not None:new.bias.copy_(old.bias[KEEP])
        branch[-1]=new
    head.nc=4;head.no=head.reg_max*4+4
    model.names=dict(enumerate(NAMES));model.yaml['nc']=4
    # Training output/optimizer metadata are intentionally not reused as new
    # four-class benchmark claims. Conversion is inference-only, not retraining.
    args={k:v for k,v in original.ckpt.get('train_args',{}).items() if k not in ['classes','resume','data','model']}
    args.update(data=str(ROOT/'data/v4_four_class_clean/data.yaml'),task='detect',model=str(destination),resume=False)
    destination.parent.mkdir(parents=True,exist_ok=True)
    provenance=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),class_mapping={'0':0,'1':1,'3':2,'4':3},
                    removed_original_class=2,retrained=False,purpose='Inference-only head projection; original features and retained weights preserved')
    torch.save(dict(model=model,epoch=-1,optimizer=None,ema=None,train_args=args,
                    date=datetime.datetime.now(datetime.timezone.utc).isoformat(),version=ultralytics.__version__,
                    license='AGPL-3.0 License (https://ultralytics.com/license)',docs='https://docs.ultralytics.com',
                    visionaid_conversion=provenance),destination)
    loaded=YOLO(str(destination));assert list(loaded.names.values())==NAMES
    # Use unfused modules on both sides for structural equality, independent of
    # class argmax and NMS, which can change when a competing class is removed.
    old=original.model.cpu().float().eval();new=loaded.model.cpu().float().eval();checks=[]
    torch.manual_seed(31)
    with torch.inference_mode():
        for kind,x in [('black',torch.zeros(1,3,640,640)),('random',torch.rand(1,3,640,640))]:
            a=old(x)[0][:,[0,1,2,3,4,5,7,8],:];b=new(x)[0]
            maximum=(a-b).abs().max().item();assert torch.allclose(a,b,rtol=1e-5,atol=1e-5),(kind,maximum)
            checks.append(dict(input=kind,max_raw_difference=maximum,anchors=a.shape[-1]))
    provenance.update(destination_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),raw_parity=checks,
                      remaining_class_weights_equal=True,post_nms_parity='Must compare on labelled data before promotion')
    return provenance


if __name__=='__main__':
    import torch
    torch.set_num_threads(4)
    result=convert(ROOT/'models/local/yolov8n_v4.pt',ROOT/'models/local/yolov8n_v4_four_class.pt')
    out=ROOT/'artifacts/four-class-cleanup';out.mkdir(exist_ok=True)
    (out/'head-conversion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
