"""Speak complete hazard alerts without blocking camera processing.

macOS uses its native speech command, avoiding the pyttsx3/PyObjC event loop
inside a worker thread. Other platforms retain pyttsx3 playback.

At most one phrase waits while another plays. A newer phrase replaces the
waiting one, a phrase that is playing is never interrupted, and a waiting phrase
that has become too old by the time its turn comes is dropped, because the
hazard it describes may no longer be in view.
"""
import logging
import subprocess
import sys
import threading
import time


logger = logging.getLogger(__name__)

MAX_PENDING_AGE_SECONDS = 1.0


class Speaker:
    def __init__(self, rate: int = 175, max_pending_age: float = MAX_PENDING_AGE_SECONDS,
                 clock=time.monotonic, start: bool = True):
        self._rate = rate
        self.max_pending_age = max_pending_age
        self._clock = clock
        self._cond = threading.Condition()
        self._pending: tuple[str, float] | None = None
        self._engine = None
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        if start:
            self._thread.start()

    def say(self, text: str):
        """Queue a phrase, replacing any phrase that is still waiting."""
        with self._cond:
            self._pending = (text, self._clock())
            self._cond.notify()

    def _next(self, block: bool = True):
        """Take the waiting phrase. Returns None if it was too old to be useful."""
        with self._cond:
            while self._pending is None:
                if not block:
                    return None
                self._cond.wait()
            text, queued_at = self._pending
            self._pending = None
        if self._clock() - queued_at > self.max_pending_age:
            self.dropped += 1
            logger.debug('Dropped stale speech: %s', text)
            return None
        return text

    def _play(self, text: str):
        try:
            if sys.platform == 'darwin':
                # Wait until the full utterance finishes. Pass text through
                # stdin so it is never interpreted as shell syntax/options.
                subprocess.run(['/usr/bin/say', '-r', str(self._rate)],
                               input=text, text=True, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            else:
                if self._engine is None:
                    import pyttsx3
                    self._engine = pyttsx3.init()
                    self._engine.setProperty('rate', self._rate)
                self._engine.say(text)
                self._engine.runAndWait()
        except Exception as exc:
            # Report playback failures and keep the next hazard audible.
            logger.error('Speech playback failed: %s',
                         getattr(exc, 'stderr', None) or str(exc), exc_info=True)

    def _run(self):
        while True:
            text = self._next()
            if text is not None:
                self._play(text)
