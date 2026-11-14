"""Speech backend regression checks; native audio calls are isolated in tests."""
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

    def popen(self, played, fail=lambda text: False, calls=None):
        """A fake `say` process: records the text, optionally fails."""
        test = self

        class Process:
            def __init__(self, args, **kwargs):
                test.assertEqual(args[:2], ['/usr/bin/say', '-r'])
                test.assertTrue(kwargs['text'])
                test.assertNotIn('shell', kwargs)
                if calls is not None:
                    calls.append(args)
                self.returncode = None
            def communicate(self, text):
                if fail(text):
                    self.returncode = 1
                    return '', 'Audio unavailable'
                played.append(text)
                self.returncode = 0
                return '', ''
            def poll(self):
                return self.returncode
        return Process

    def test_mac_plays_whole_phrases_serially_through_native_speech(self):
        speaker = self.speaker()
        played, calls = [], []
        old_engine_speech = []
        with patch.object(sys, 'platform', 'darwin'), \
             patch('subprocess.Popen', self.popen(played, calls=calls)), \
             patch('pyttsx3.init', return_value=self.engine(old_engine_speech)):
            for phrase in ('Warning, Stair ahead', 'Object detected on your left'):
                speaker.say(phrase)
                self.drain(speaker)
        self.assertEqual(played, ['Warning, Stair ahead', 'Object detected on your left'])
        self.assertEqual(calls[0], ['/usr/bin/say', '-r', '180'])  # default rate (U5)
        self.assertEqual(old_engine_speech, [])

    def test_mac_playback_failure_is_reported_and_does_not_kill_next_alert(self):
        speaker = self.speaker()
        played = []
        with patch.object(sys, 'platform', 'darwin'), \
             patch('subprocess.Popen', self.popen(played, fail=lambda text: text.startswith('Warning'))), \
             patch('pyttsx3.init', return_value=self.engine([])):
            with self.assertLogs('tts', level='ERROR') as logs:
                for phrase in ('Warning, Stair ahead', 'Object detected ahead'):
                    speaker.say(phrase)
                    self.drain(speaker)
        self.assertEqual(played, ['Object detected ahead'])
        self.assertIn('Audio unavailable', '\n'.join(logs.output))
        self.assertIsNone(speaker.error)  # the later success cleared it

    def test_failure_sets_error_for_the_page(self):
        speaker = self.speaker()
        with patch.object(sys, 'platform', 'darwin'), \
             patch('subprocess.Popen', side_effect=FileNotFoundError('/usr/bin/say')), \
             self.assertLogs('tts', level='ERROR'):
            speaker.say('Warning, Stair ahead')
            self.drain(speaker)
        self.assertIn('say', speaker.error)

    def test_rate_change_applies_to_the_next_phrase(self):
        speaker = self.speaker()
        played, calls = [], []
        with patch.object(sys, 'platform', 'darwin'), patch('subprocess.Popen', self.popen(played, calls=calls)):
            speaker.say('one')
            self.drain(speaker)
            speaker.rate = 240
            speaker.say('two')
            self.drain(speaker)
        self.assertEqual([c[2] for c in calls], ['180', '240'])

    def test_stop_ends_the_playing_phrase_and_clears_the_waiting_one(self):
        import threading
        started, ended = threading.Event(), threading.Event()

        class Process:
            def __init__(self, args, **kwargs):
                self.returncode = None
                self.done = threading.Event()
            def communicate(self, text):
                started.set()
                self.done.wait(5)
                ended.set()
                return '', ''
            def poll(self):
                return self.returncode
            def terminate(self):
                self.returncode = -15
                self.done.set()

        speaker = Speaker(start=False)
        with patch.object(sys, 'platform', 'darwin'), patch('subprocess.Popen', Process):
            worker = threading.Thread(target=speaker._play, args=('Warning, Stair ahead',))
            worker.start()
            self.assertTrue(started.wait(2))
            speaker.say('Caution, People ahead')
            speaker.stop()
            self.assertTrue(ended.wait(2))
            worker.join(2)
        self.assertIsNone(speaker._next(block=False))  # waiting phrase cleared
        self.assertIsNone(speaker.error)  # a stopped phrase is not a failure

    def test_other_platforms_keep_pyttsx3_hazard_speech(self):
        speaker = self.speaker()
        spoken = []
        with patch.object(sys, 'platform', 'linux'), \
             patch('pyttsx3.init', return_value=self.engine(spoken)), \
             patch('subprocess.Popen', side_effect=AssertionError('macOS backend used on Linux')):
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
