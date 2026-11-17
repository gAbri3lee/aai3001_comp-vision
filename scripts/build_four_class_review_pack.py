"""Export original/corrected boxes for independent team annotation sign-off."""
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAMES=['Escalator','People','Stair','Wet Floor Sign']


def panel(image,rows,old_count,caption):
    boxes=[]
    for index,(cid,cx,cy,width,height) in enumerate(rows):
        color='#22c55e' if index>=old_count else '#fbbf24'
        x=(cx-width/2)*640;y=(cy-height/2)*640
        boxes.append(f'<rect x="{x}" y="{y}" width="{width*640}" height="{height*640}" stroke="{color}" fill="none" stroke-width="3"/>')
        boxes.append(f'<text x="{x+2}" y="{max(15,y-4)}" fill="{color}" stroke="#000" stroke-width=".5" font-size="15">{html.escape(NAMES[int(cid)])}</text>')
    return f'<figure><figcaption>{caption}</figcaption><svg role="img" aria-label="{caption}" viewBox="0 0 640 640"><image width="640" height="640" href="data:image/jpeg;base64,{image}"/>{"".join(boxes)}</svg></figure>'


if __name__=='__main__':
    review=json.loads((ROOT/'annotations/four_class_training_review.json').read_text())
    output=ROOT/'artifacts/four-class-study/team-review';output.mkdir(parents=True,exist_ok=True)
    cards=[];records=[]
    for stem,item in review['repairs'].items():
        source=ROOT/'data/v4_four_class_clean/train/images'/(stem+'.jpg')
        pixels=source.read_bytes();assert hashlib.sha256(pixels).hexdigest()==item['sha256']
        image=base64.b64encode(pixels).decode();old=item['original_labels'];new=item['replacement_labels']
        added=len(new)-len(old);assert added>0 and new[:len(old)]==old
        cards.append(f'<section><h2>{html.escape(item["review_id"])} — {added} added box(es)</h2><p>{html.escape(stem)}</p><div class="pair">{panel(image,old,len(old),"Original labels")}{panel(image,new,len(old),"Revised labels: green boxes added")}</div></section>')
        records.append(dict(review_id=item['review_id'],stem=stem,sha256=item['sha256'],added_boxes=added,reviewer='',decision='PENDING',notes=''))
    markup='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Four-class annotation sign-off</title><style>body{font:16px system-ui;max-width:1380px;margin:24px auto;padding:0 16px;background:#101827;color:#eef2ff}h1{font-size:28px}section{padding:16px 0;border-top:1px solid #475569}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}figcaption{margin:8px 0}svg{width:100%;height:auto}p{overflow-wrap:anywhere}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style><h1>Independent team review required</h1><p>AI proposed these training-label additions. No human sign-off has been recorded. Yellow boxes are unchanged; green boxes are additions. Check class, box extent, occlusion and every other visible retained-class target. Use the CSV to approve or request corrections; do not infer approval from viewing this page.</p>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(markup,encoding='utf-8')
    # Preserve a team's existing decisions on regeneration.
    csv_path=output/'signoff.csv'
    if not csv_path.exists():
        with csv_path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print('review_frames',len(records),'added_boxes',sum(r['added_boxes'] for r in records))
