"""Finish authorised experiments serially; stop on failures instead of claiming success."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'artifacts/mall-whole-scene-study'
    while len(json.loads((out/'status.json').read_text())['completed'])<9:time.sleep(2)
    for script in ['evaluate_mall_whole_scene.py','mall_fresh_obstacle_test.py','mall_supplement_validation.py','mall_video_warning_check.py']:
        print('STAGE',script,flush=True);subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
    video=out/'repeated-vivocity-60s.mp4'
    subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-stream_loop','-1','-i',str(ROOT/'artifacts/online-source-uploads-isolated-final/vivocity-source-converted.mp4'),'-t','60','-c','copy',str(video)],check=True)
    for script,name in [('benchmark_app.py','current'),('mall_runtime_profile.py','research')]:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),'--source',str(video),'--general','on','--frames','40','--threads','4','--output',str(out/(name+'-runtime.json'))],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','pytest','-q','--basetemp',str(ROOT.parent/'tmp/mall-study-final-tests')],cwd=ROOT,check=True)
    (out/'all_checks_complete.json').write_text(json.dumps(dict(completed=True,physical_tests='deferred by user; setup unavailable',no_automatic_deployment=True)))
    print('ALL_CHECKS_COMPLETE',flush=True)


if __name__=='__main__':main()
