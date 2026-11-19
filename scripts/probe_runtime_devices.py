"""Probe the laptop camera without saving frames, and exercise real native speech."""
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    import cv2
    import numpy as np
    report = {'camera': {}, 'speech': {}}
    cap = cv2.VideoCapture(0)
    try:
        report['camera']['opened'] = cap.isOpened()
        frames, times, variations = [], [], []
        for _ in range(20 if cap.isOpened() else 0):
            start = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                break
            times.append(time.monotonic()-start)
            frames.append(frame.shape)
            variations.append(float(np.std(frame)))
        report['camera'].update(frames=len(frames), shape=frames[-1] if frames else None,
            median_read_ms=1000*float(np.median(times)) if times else None,
            median_pixel_std=float(np.median(variations)) if variations else None,
            frames_saved=False, measured_distances_available=False,
            note='Device 0 is a provisional laptop-camera candidate, not a calibrated distance trial.')
    finally:
        cap.release()
    from tts import Speaker
    import pyttsx3
    real_init = pyttsx3.init
    events = []
    finished, began = threading.Event(), threading.Event()
    def init(*args, **kwargs):
        engine = real_init(*args, **kwargs)
        engine.connect('started-utterance', lambda name: (events.append(['started',time.monotonic()]),began.set()))
        engine.connect('finished-utterance', lambda name, completed: (events.append(['finished',time.monotonic(),completed]),finished.set()))
        return engine
    with patch('pyttsx3.init', side_effect=init):
        speaker = Speaker()
        queued = time.monotonic()
        speaker.say('VisionAid voice test')
        completed = finished.wait(12)
        report['speech'] = dict(completion_callback=completed, events=events,
            queue_to_start_ms=1000*(events[0][1]-queued) if events else None,
            error=speaker.error, audibility_confirmed_by_human=False)
        finished.clear()
        began.clear()
        speaker.say('VisionAid second voice test')
        report['speech']['second_completion_callback'] = finished.wait(12)
        finished.clear()
        began.clear()
        speaker.say('VisionAid interruption test. This longer phrase should stop before it finishes. '
                    'This is a test of stopping the speech when the user mutes alerts.')
        report['speech']['interrupt_started'] = began.wait(5)
        stopping = time.monotonic()
        speaker.stop()
        report['speech']['interrupt_callback'] = finished.wait(3)
        report['speech']['stop_to_callback_ms'] = 1000 * (time.monotonic()-stopping)
        finished.clear()
        speaker.say('VisionAid speech recovered')
        report['speech']['recovery_callback'] = finished.wait(12)
        speaker.close()
        report['speech']['worker_closed'] = not speaker._thread.is_alive()
        report['speech']['final_error'] = speaker.error
        outcomes = [event[2] for event in events if event[0] == 'finished']
        report['speech']['expected_completion_outcomes'] = outcomes == [True, True, False, True]
        starts = [event[1] for event in events if event[0] == 'started']
        ends = [event[1] for event in events if event[0] == 'finished']
        report['speech']['utterance_durations_ms'] = [1000*(end-start) for start,end in zip(starts,ends)]
    out = Path('artifacts/system-validation/devices.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['speech']['expected_completion_outcomes']:
        raise SystemExit('Native speech completion/interruption/recovery did not match expectations')


if __name__ == '__main__':
    main()
