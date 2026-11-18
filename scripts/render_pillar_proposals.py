"""Render TRAIN-only proposals for visual adjudication; labels remain untouched."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from path_utils import native_path
from golden import iou
from config import CLASS_NAMES


def main():
    from PIL import Image,ImageDraw,ImageOps
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposals',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/v4_scene_split')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=120)
    args=parser.parse_args()
    payload=args.proposals.read_bytes()
    audit=json.loads(payload)
    if audit.get('test_used') or audit.get('validation_used'):
        raise ValueError('Training-only review required')
    proposals=sorted(audit['proposals'],key=lambda p:(
        -int(p['source']=='world' and p['prompt'] in ('pillar','column')),
        p['other_label_iou']>=.3,-p['confidence'],p['image']))
    chosen=[]
    scenes=Counter()
    for proposal in proposals:
        if scenes[proposal['scene']]>=2:
            continue
        if any(p['image']==proposal['image'] and iou(p['box_xyxy'],proposal['box_xyxy'])>=.7 for p in chosen):
            continue
        chosen.append(dict(proposal,review_id=f'A{len(chosen):04d}',decision='pending',reviewer=''))
        scenes[proposal['scene']]+=1
        if len(chosen)>=args.limit:
            break
    args.output.mkdir(parents=True,exist_ok=True)
    record=dict(proposals_sha256=hashlib.sha256(payload).hexdigest(),records=chosen,
        note='Red: proposed box. Green: existing labels. All decisions initially pending; no labels changed.')
    (args.output/'review.json').write_text(json.dumps(record,indent=2))
    for start in range(0,len(chosen),12):
        page=Image.new('RGB',(1500,2160),'white')
        text=ImageDraw.Draw(page)
        for index,row in enumerate(chosen[start:start+12]):
            with Image.open(native_path(args.dataset/'train/images'/row['image'])) as source:
                frame=source.convert('RGB')
            draw=ImageDraw.Draw(frame)
            w,h=frame.size
            label=native_path(args.dataset/'train/labels'/(Path(row['image']).stem+'.txt'))
            for line in label.read_text().splitlines() if label.exists() else []:
                cid,x,y,bw,bh=map(float,line.split())
                box=((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)
                draw.rectangle(box,outline='lime',width=3)
                draw.text((box[0]+2,box[1]+2),CLASS_NAMES[int(cid)],fill='lime')
            draw.rectangle(row['box_xyxy'],outline='red',width=4)
            tile=ImageOps.contain(frame,(492,480))
            x,y=(index%3)*500+4,(index//3)*540+4
            page.paste(tile,(x,y))
            text.text((x,y+483),f"{row['review_id']} {row['source']} {row['prompt']} {row['confidence']:.2f}",fill='black')
            text.text((x,y+500),row['scene'][:55],fill='black')
            text.text((x,y+517),f"other-label IoU {row['other_label_iou']:.2f} / PENDING",fill='black')
        page.save(args.output/f'proposals_{start//12:02d}.jpg',quality=92)
    print(json.dumps(dict(selected=len(chosen),recorded_scenes=len(scenes))))


if __name__=='__main__':
    main()
