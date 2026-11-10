"""Speak complete hazard alerts without blocking camera processing.

macOS uses its native speech command, avoiding the pyttsx3/PyObjC event loop
inside a worker thread. Other platforms retain pyttsx3 playback.

At most one phrase waits while another plays. A newer phrase replaces the
waiting one, a phrase that is playing is never interrupted by a newer alert, and
a waiting phrase that has become too old by the time its turn comes is dropped,
because the hazard it describes may no longer be in view.

`stop()` is the one exception: it ends the phrase that is playing and clears the
waiting one (used by mute and pause). `rate` (words per minute) applies to the
next phrase. `error` is None while speech works. It holds the reason when the
speech command is missing or the last phrase failed, so the page can say alerts
are on screen only.
"""
import os
import logging
import subprocess
import sys
import threading
import time


logger = logging.getLogger(__name__)

MAX_PENDING_AGE_SECONDS = 1.0
DEFAULT_RATE = 180  # Words per minute (decision U5).
SAY = '/usr/bin/say'


class Speaker:
    def __init__(self, rate: int = DEFAULT_RATE, max_pending_age: float = MAX_PENDING_AGE_SECONDS,
                 clock=time.monotonic, start: bool = True):
        self.rate = rate
        self.max_pending_age = max_pending_age
        self._clock = clock
        self._cond = threading.Condition()
        self._pending: tuple[str, float] | None = None
        self._process = None
        self._stopped = False
        self._engine = None
        self.dropped = 0
        self.error = None
        if sys.platform == 'darwin' and not os.path.exists(SAY):
            self.error = f'Speech command {SAY} not found'
        self._thread = threading.Thread(target=self._run, daemon=True)
        if start:
            self._thread.start()

    def say(self, text: str):
        """Queue a phrase, replacing any phrase that is still waiting."""
        with self._cond:
            self._pending = (text, self._clock())
            self._cond.notify()

    def stop(self):
        """End the phrase that is playing now and forget the waiting one."""
        with self._cond:
            self._pending = None
            process, self._stopped = self._process, True
        if process is not None and process.poll() is None:
            process.terminate()
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                logger.debug('pyttsx3 stop failed', exc_info=True)

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
                args = [SAY, '-r', str(int(self.rate))]
                process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, text=True)
                with self._cond:
                    self._process, self._stopped = process, False
                try:
                    _, stderr = process.communicate(text)
                finally:
                    with self._cond:
                        self._process = None
                        stopped = self._stopped
                if process.returncode != 0 and not stopped:
                    raise subprocess.CalledProcessError(process.returncode, args, stderr=stderr)
            else:
                if self._engine is None:
                    import pyttsx3
                    self._engine = pyttsx3.init()
                self._engine.setProperty('rate', int(self.rate))
                self._engine.say(text)
                self._engine.runAndWait()
            self.error = None
        except Exception as exc:
            self.error = str(getattr(exc, 'stderr', None) or exc) or type(exc).__name__
            # Report playback failures and keep the next hazard audible.
            logger.error('Speech playback failed: %s',
                         getattr(exc, 'stderr', None) or str(exc), exc_info=True)

    def _run(self):
        while True:
            text = self._next()
            if text is not None:
                self._play(text)
