"""
Text-to-speech output stage. Runs on a background thread so speaking never
blocks the video/detection loop (see CLAUDE.md pipeline stage 4).
"""
import queue
import threading

import pyttsx3


class Speaker:
    def __init__(self, rate: int = 175):
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._rate = rate
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        while True:
            text = self._queue.get()
            engine.say(text)
            engine.runAndWait()

    def say(self, text: str):
        # Drop new alerts while one is queued, so speech doesn't lag behind
        # what's actually on screen.
        if self._queue.empty():
            self._queue.put(text)
