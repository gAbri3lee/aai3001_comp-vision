"""Speak complete hazard alerts without blocking camera processing.

macOS uses its native speech command, avoiding the pyttsx3/PyObjC event loop
inside a worker thread. Other platforms retain pyttsx3 playback.
"""
import logging
import queue
import subprocess
import sys
import threading


logger = logging.getLogger(__name__)


class Speaker:
    def __init__(self, rate: int = 175):
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=1)
        self._rate = rate
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        engine = None
        while True:
            text = self._queue.get()
            try:
                if sys.platform == 'darwin':
                    # Wait until the full utterance finishes. Pass text through
                    # stdin so it is never interpreted as shell syntax/options.
                    subprocess.run(['/usr/bin/say', '-r', str(self._rate)],
                                   input=text, text=True, check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                else:
                    if engine is None:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.setProperty('rate', self._rate)
                    engine.say(text)
                    engine.runAndWait()
            except Exception as exc:
                # Report playback failures and keep the next hazard audible.
                logger.error('Speech playback failed: %s',
                             getattr(exc, 'stderr', None) or str(exc), exc_info=True)
            finally:
                self._queue.task_done()

    def say(self, text: str):
        # Keep at most one pending alert. Never interrupt the current phrase.
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass
