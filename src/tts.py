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
        self._closed = False
        self._epoch = 0
        self._dequeued_epoch = None
        self._windows_loop = False
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
            if self._closed:
                return
            self._pending = (text, self._clock())
            self._cond.notify()

    def stop(self):
        """End the phrase that is playing now and forget the waiting one."""
        with self._cond:
            self._pending = None
            self._epoch += 1
            process, self._stopped = self._process, True
        if process is not None and process.poll() is None:
            process.terminate()
        if self._engine is not None and sys.platform != 'win32':
            try:
                self._engine.stop()
            except Exception:
                logger.debug('pyttsx3 stop failed', exc_info=True)

    def close(self):
        """Stop playback and release the idle worker during app shutdown."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self.stop()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)

    def _next(self, block: bool = True):
        """Take the waiting phrase. Returns None if it was too old to be useful."""
        with self._cond:
            while self._pending is None:
                if self._closed or not block:
                    return None
                self._cond.wait()
            text, queued_at = self._pending
            self._pending = None
            self._dequeued_epoch = self._epoch
        if self._clock() - queued_at > self.max_pending_age:
            self.dropped += 1
            logger.debug('Dropped stale speech: %s', text)
            return None
        return text

    def _play(self, text: str):
        try:
            with self._cond:
                if self._closed or (self._dequeued_epoch is not None and self._dequeued_epoch != self._epoch):
                    return
                self._stopped = False
            if sys.platform == 'darwin':
                # Wait until the full utterance finishes. Pass text through
                # stdin so it is never interpreted as shell syntax/options.
                args = [SAY, '-r', str(int(self.rate))]
                with self._cond:
                    if self._stopped or self._closed:
                        return
                    process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                               stderr=subprocess.PIPE, text=True)
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
                if sys.platform == 'win32':
                    self._play_windows(text)
                else:
                    self._engine.say(text)
                    self._engine.runAndWait()
            self.error = None
        except Exception as exc:
            self.error = str(getattr(exc, 'stderr', None) or exc) or type(exc).__name__
            # Report playback failures and keep the next hazard audible.
            logger.error('Speech playback failed: %s',
                         getattr(exc, 'stderr', None) or str(exc), exc_info=True)

    def _play_windows(self, text):
        """Keep SAPI and its COM event pump on the owning worker thread.

        A persistent external loop also avoids runAndWait/endLoop cancelling
        the next utterance when the previous completion callback fires.
        """
        if not self._windows_loop:
            self._engine.startLoop(False)
            self._windows_loop = True
        done = threading.Event()
        outcomes = []
        token = self._engine.connect('finished-utterance',
            lambda name, completed: (outcomes.append(completed), done.set()))
        try:
            with self._cond:
                if self._stopped or self._closed:
                    return
                self._engine.say(text)
            deadline = time.monotonic() + 60
            stop_sent = False
            while not done.is_set():
                with self._cond:
                    stopping = self._stopped or self._closed
                if stopping and not stop_sent:
                    self._engine.stop()
                    stop_sent = True
                self._engine.iterate()
                if time.monotonic() > deadline:
                    self._engine.stop()
                    raise TimeoutError('Windows speech did not finish within 60 seconds')
                done.wait(.01)
            if outcomes and not outcomes[-1] and not stop_sent:
                raise RuntimeError('Windows speech ended before completing the phrase')
            if stop_sent:
                # SAPI purge emits an additional empty-stream completion.
                # Drain it before a new phrase can inherit that old callback.
                drain_until = time.monotonic() + .2
                while time.monotonic() < drain_until:
                    self._engine.iterate()
                    time.sleep(.01)
        finally:
            self._engine.disconnect(token)
            if self._stopped:
                # A purged SAPI stream can leave a completion queued until the
                # next Speak call. Retire this engine after mute instead of
                # letting that event complete the next hazard prematurely.
                self._engine.endLoop()
                self._windows_loop = False
                self._engine = None

    def _run(self):
        while True:
            text = self._next()
            with self._cond:
                if self._closed:
                    if self._windows_loop:
                        self._engine.endLoop()
                    return
            if text is not None:
                self._play(text)
