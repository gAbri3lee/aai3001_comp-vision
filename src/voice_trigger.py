"""
Voice-triggered on-demand query stage (CLAUDE.md pipeline stage 2,
secondary track). Listens continuously in the background for a spoken
trigger phrase (e.g. "what is this") and flags the main loop to run OCR.

This is a demo approximation, not a low-latency on-device wake-word engine:
it uses the free Google Web Speech API via SpeechRecognition, so it needs
internet access and has real network round-trip latency per utterance.
"""
import threading

import speech_recognition as sr

DEFAULT_TRIGGER_PHRASES = ("vision scan",)


class VoiceTrigger:
    def __init__(self, trigger_phrases=DEFAULT_TRIGGER_PHRASES, mic_index=None):
        self.trigger_phrases = trigger_phrases
        self._recognizer = sr.Recognizer()
        self._mic = sr.Microphone(device_index=mic_index)
        self._triggered = threading.Event()
        self._stop = threading.Event()

        with self._mic as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=1)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                with self._mic as source:
                    audio = self._recognizer.listen(source, timeout=3, phrase_time_limit=4)
                text = self._recognizer.recognize_google(audio).lower()
                print(f"[voice] heard: {text}")
                if any(phrase in text for phrase in self.trigger_phrases):
                    self._triggered.set()
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"[voice] speech recognition service error: {e}")
            except Exception as e:
                print(f"[voice] error: {e}")

    def consume_trigger(self) -> bool:
        """Returns True once if a trigger phrase was heard since the last check."""
        if self._triggered.is_set():
            self._triggered.clear()
            return True
        return False

    def stop(self):
        self._stop.set()
