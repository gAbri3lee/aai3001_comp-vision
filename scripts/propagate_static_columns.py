"""Conservative TRAIN-only column propagation; outputs await visual approval.

Uses manually drawn reference columns, not detector boxes. Requires near-identity
ORB registration and photometric agreement. Nonmatching frames are excluded
from the proposed next training schedule, never silently declared background.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from path_utils import native_path, read_bgr


def main():
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageOps
    cv2.setNumThreads(2)
    dataset=ROOT/'data/v4_scene_split'
    output=ROOT/'artifacts/pillar-annotation-adjudication/propagation'
    output.mkdir(parents=True,exist_ok=True)
    patch=json.loads((ROOT/'configs/pillar_label_repair_1.json').read_text())
    references=[r for r in patch['records'] if r.get('review_reference') in ('A0004','A0009','A0012')]
    manual={r['image'] for r in patch['records']}
    manifest=list(csv.DictReader((dataset/'split_manifest.csv').read_text().splitlines()))
    scenes={r['stem']:r['scene'] for r in manifest if r['split']=='train'}
    scene_names={r['scene'] for r in references}
    images=[p for p in sorted((dataset/'train/images').iterdir()) if scenes[p.stem] in scene_names]
    orb=cv2.ORB_create(nfeatures=1500)
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    refs=[]
    for reference in references:
        frame=read_bgr(dataset/'train/images'/reference['image'])
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        points,descriptors=orb.detectAndCompute(gray,None)
        refs.append((reference,frame,points,descriptors))
    proposals=[]
    excluded=[]
    for path in images:
        if path.name in manual:
            continue
        frame=read_bgr(path)
        h,w=frame.shape[:2]
        keypoints,descriptors=orb.detectAndCompute(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),None)
        possibilities=[]
        for reference,source,ref_points,ref_descriptors in refs:
            if reference['scene']!=scenes[path.stem] or descriptors is None:
                continue
            pairs=matcher.knnMatch(ref_descriptors,descriptors,k=2)
            good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance < .75*pair[1].distance]
            if len(good)<30:
                continue
            a=np.float32([ref_points[m.queryIdx].pt for m in good])
            b=np.float32([keypoints[m.trainIdx].pt for m in good])
            matrix,inliers=cv2.findHomography(a,b,cv2.RANSAC,2.)
            if matrix is None or inliers.mean()<.6:
                continue
            corners=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]).reshape(-1,1,2)
            shifted=cv2.perspectiveTransform(corners,matrix)
            drift=float(np.linalg.norm(shifted-corners,axis=2).max())
            if drift>4:
                continue
            warped=cv2.warpPerspective(source,matrix,(w,h))
            boxes=[]
            fractions=[]
            for box in reference['add_boxes']:
                if box['class_id']!=2:
                    continue
                x1,y1,x2,y2=box['xyxy']
                vertices=np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]).reshape(-1,1,2)
                transformed=cv2.perspectiveTransform(vertices,matrix).reshape(-1,2)
                x1,y1=np.maximum(transformed.min(0),[0,0])
                x2,y2=np.minimum(transformed.max(0),[w,h])
                if x2-x1<5 or y2-y1<5:
                    break
                left,top,right,bottom=map(int,(x1,y1,x2,y2))
                difference=np.abs(frame[top:bottom,left:right].astype(float)-warped[top:bottom,left:right].astype(float)).mean(2)
                fraction=float((difference<45).mean())
                if fraction<.65:
                    break
                fractions.append(fraction)
                boxes.append(dict(class_id=2,xyxy=[round(float(v),2) for v in (x1,y1,x2,y2)]))
            expected=sum(b['class_id']==2 for b in reference['add_boxes'])
            if len(boxes)!=expected:
                continue
            possibilities.append((sum(fractions)/len(fractions),reference,boxes,drift,float(inliers.mean())))
        if not possibilities:
            excluded.append(path.name)
            continue
        score,reference,boxes,drift,inlier_fraction=max(possibilities,key=lambda p:p[0])
        label=native_path(dataset/'train/labels'/(path.stem+'.txt'))
        data=label.read_bytes() if label.exists() else b''
        proposals.append(dict(image=path.name,scene=scenes[path.stem],
            image_sha256=hashlib.sha256(native_path(path).read_bytes()).hexdigest(),
            original_label_sha256=hashlib.sha256(data).hexdigest(),
            canonical_label_sha256=hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest(),
            reviewed_full_image=False,reviewer='',add_boxes=boxes,remove_lines=[],
            reference=reference['review_reference'],registration_corner_drift_px=drift,
            inlier_fraction=inlier_fraction,photometric_agreement=score,
            status='PENDING whole-frame overlay review; not approved ground truth'))
    report=dict(records=proposals,excluded_train_images=excluded,
        scenes=sorted(scene_names),test_used=False,validation_used=False,
        policy=dict(max_corner_drift_px=4,min_inlier_fraction=.6,min_matches=30,
                    min_photometric_fraction=.65,max_pixel_difference=45),
        note='References manually annotated. Registration only proposes propagation; visual approval still required.')
    (output/'candidates.json').write_text(json.dumps(report,indent=2))
    for start in range(0,len(proposals),24):
        page=Image.new('RGB',(1600,1900),'white')
        draw=ImageDraw.Draw(page)
        for index,row in enumerate(proposals[start:start+24]):
            with Image.open(native_path(dataset/'train/images'/row['image'])) as source:
                frame=source.convert('RGB')
            fd=ImageDraw.Draw(frame)
            for box in row['add_boxes']:
                fd.rectangle(box['xyxy'],outline='red',width=3)
            tile=ImageOps.contain(frame,(392,280))
            x,y=(index%4)*400+4,(index//4)*315+4
            page.paste(tile,(x,y))
            draw.text((x,y+282),f"C{start+index:03d} ref={row['reference']} shift={row['registration_corner_drift_px']:.1f}",fill='black')
            draw.text((x,y+296),f"agreement={row['photometric_agreement']:.2f} PENDING",fill='black')
        page.save(output/f'propagation_{start//24:02d}.jpg',quality=92)
    print(json.dumps(dict(proposed_images=len(proposals),excluded_images=len(excluded),source_images=len(images))))


if __name__=='__main__':
    main()
