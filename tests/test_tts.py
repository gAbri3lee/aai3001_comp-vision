"""Speech backend regression checks; native audio calls are isolated in tests."""
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tts import Speaker


class FakeClock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class SpeakerTests(unittest.TestCase):
    def speaker(self, clock=None):
        return Speaker(clock=clock or FakeClock(), start=False)

    def engine(self, spoken):
        return SimpleNamespace(setProperty=lambda *args: None,
                               say=spoken.append, runAndWait=lambda: None)

    def drain(self, speaker):
        """Play everything currently waiting, as the worker thread would."""
        while (text := speaker._next(block=False)) is not None:
            speaker._play(text)

    def test_mac_plays_whole_phrases_serially_through_native_speech(self):
        speaker = self.speaker()
        played = []
        old_engine_speech = []
        def play(args, **kwargs):
            self.assertEqual(args, ['/usr/bin/say', '-r', '175'])
            self.assertTrue(kwargs['check'])
            self.assertTrue(kwargs['text'])
            self.assertNotIn('shell', kwargs)
            played.append(kwargs['input'])
        with patch.object(sys, 'platform', 'darwin'), \
             patch('subprocess.run', side_effect=play), \
             patch('pyttsx3.init', return_value=self.engine(old_engine_speech)):
            for phrase in ('Warning, Stair ahead', 'Object detected on your left'):
                speaker.say(phrase)
                self.drain(speaker)
        self.assertEqual(played, ['Warning, Stair ahead', 'Object detected on your left'])
        self.assertEqual(old_engine_speech, [])

    def test_mac_playback_failure_is_reported_and_does_not_kill_next_alert(self):
        speaker = self.speaker()
        played = []
        def play(args, **kwargs):
            if kwargs['input'].startswith('Warning'):
                raise subprocess.CalledProcessError(1, args, stderr='Audio unavailable')
            played.append(kwargs['input'])
        with patch.object(sys, 'platform', 'darwin'), \
             patch('subprocess.run', side_effect=play), \
             patch('pyttsx3.init', return_value=self.engine([])):
            with self.assertLogs('tts', level='ERROR') as logs:
                for phrase in ('Warning, Stair ahead', 'Object detected ahead'):
                    speaker.say(phrase)
                    self.drain(speaker)
        self.assertEqual(played, ['Object detected ahead'])
        self.assertIn('Audio unavailable', '\n'.join(logs.output))

    def test_other_platforms_keep_pyttsx3_hazard_speech(self):
        speaker = self.speaker()
        spoken = []
        with patch.object(sys, 'platform', 'linux'), \
             patch('pyttsx3.init', return_value=self.engine(spoken)), \
             patch('subprocess.run', side_effect=AssertionError('macOS backend used on Linux')):
            speaker.say('Warning, Stair ahead')
            self.drain(speaker)
        self.assertEqual(spoken, ['Warning, Stair ahead'])

    def test_newest_message_replaces_the_pending_one(self):
        speaker = self.speaker()
        speaker.say('Warning, Stair ahead')
        speaker.say('Caution, People ahead')
        self.assertEqual(speaker._next(block=False), 'Caution, People ahead')
        self.assertIsNone(speaker._next(block=False))

    def test_stale_pending_message_is_dropped(self):
        clock = FakeClock()
        speaker = self.speaker(clock)
        speaker.say('Warning, Stair ahead')
        clock.now = 1.5  # e.g. a long phrase was playing meanwhile
        self.assertIsNone(speaker._next(block=False))
        self.assertEqual(speaker.dropped, 1)
        speaker.say('Warning, Stair ahead')
        clock.now = 2.0
        self.assertEqual(speaker._next(block=False), 'Warning, Stair ahead')

    def test_worker_thread_plays_without_blocking_the_caller(self):
        import threading
        done = threading.Event()
        played = []
        speaker = Speaker(start=False)
        speaker._play = lambda text: (played.append(text), done.set())
        speaker._thread.start()
        speaker.say('Warning, Stair ahead')
        self.assertTrue(done.wait(2))
        self.assertEqual(played, ['Warning, Stair ahead'])


if __name__ == '__main__':
    unittest.main()
