"""Count column-warning events on the reviewed short escalator clip; no range claims."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from config import add_detector_arguments,detector_settings
from decision_engine import AlertPolicy
from scene_detector import SceneDetector
from mall_runtime_profile import ColumnSupplement


def main():
    import argparse,cv2,torch
    torch.set_num_threads(4);p=argparse.ArgumentParser();add_detector_arguments(p);settings=detector_settings(p.parse_args([]))
    out=ROOT/'artifacts/mall-whole-scene-study';e=json.loads((out/'evaluation.json').read_text());weights=out/e['selected_arm']/'seed0/weights/last.pt'
    video=ROOT/'artifacts/online-source-uploads-isolated-final/vivocity-source-converted.mp4';reports={}
    for name,model in [('current',SceneDetector(settings)),('research_supplement',ColumnSupplement(settings,weights))]:
        capture=cv2.VideoCapture(str(video));fps=capture.get(cv2.CAP_PROP_FPS);frames=0;events=[];gated_frames=0;policy=AlertPolicy(settings['alert_gate'],settings['severity'])
        while True:
            ok,frame=capture.read()
            if not ok:break
            if frames%2==0:
                batch=model.predict(frame);alerts=policy.process(batch.detections,(frame.shape[1],frame.shape[0]),now=frames/fps)
                gated_frames+=any(h.gated and h.label=='Pillar' for h in alerts.hazards)
                if alerts.speak and alerts.selected.label=='Pillar':events.append(dict(at_seconds=frames/fps,phrase=alerts.speak))
            frames+=1
        capture.release();duration=frames/fps
        reports[name]=dict(duration_seconds=duration,sampled_frames=(frames+1)//2,column_warning_events=events,
            gated_column_frames=gated_frames,events_per_minute=len(events)*60/duration)
    (out/'video-warning-check.json').write_text(json.dumps(dict(results=reports,
        limitation='Only 9.47 seconds, repeated source from prior work. No clear large structural columns observed in visual review. Normalised rate is not a population false-warning estimate; audio is not played.'),indent=2))
    print(reports,flush=True)


if __name__=='__main__':main()
