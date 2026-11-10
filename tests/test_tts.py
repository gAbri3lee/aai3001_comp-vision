"""Speech backend regression checks; native audio calls are isolated in tests."""
import queue
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tts import Speaker


class WorkerFinished(BaseException):
    pass


class FiniteQueue(queue.Queue):
    def get(self, *args, **kwargs):
        try:
            return super().get(block=False)
        except queue.Empty:
            raise WorkerFinished()


class SpeakerTests(unittest.TestCase):
    def speaker_with(self, *phrases):
        speaker = Speaker.__new__(Speaker)
        speaker._rate = 175
        speaker._queue = FiniteQueue()
        for phrase in phrases:
            speaker._queue.put(phrase)
        return speaker

    def engine(self, spoken):
        return SimpleNamespace(setProperty=lambda *args: None,
                               say=spoken.append, runAndWait=lambda: None)

    def test_mac_plays_whole_phrases_serially_through_native_speech(self):
        speaker = self.speaker_with('Warning, Stair ahead', 'Object detected on your left')
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
            with self.assertRaises(WorkerFinished):
                speaker._run()
        self.assertEqual(played, ['Warning, Stair ahead', 'Object detected on your left'])
        self.assertEqual(old_engine_speech, [])

    def test_mac_playback_failure_is_reported_and_does_not_kill_next_alert(self):
        speaker = self.speaker_with('Warning, Stair ahead', 'Object detected ahead')
        played = []
        def play(args, **kwargs):
            if kwargs['input'].startswith('Warning'):
                raise subprocess.CalledProcessError(1, args, stderr='Audio unavailable')
            played.append(kwargs['input'])
        with patch.object(sys, 'platform', 'darwin'), \
             patch('subprocess.run', side_effect=play), \
             patch('pyttsx3.init', return_value=self.engine([])):
            with self.assertLogs('tts', level='ERROR') as logs:
                with self.assertRaises(WorkerFinished):
                    speaker._run()
        self.assertEqual(played, ['Object detected ahead'])
        self.assertIn('Audio unavailable', '\n'.join(logs.output))

    def test_other_platforms_keep_pyttsx3_hazard_speech(self):
        speaker = self.speaker_with('Warning, Stair ahead')
        spoken = []
        with patch.object(sys, 'platform', 'linux'), \
             patch('pyttsx3.init', return_value=self.engine(spoken)), \
             patch('subprocess.run', side_effect=AssertionError('macOS backend used on Linux')):
            with self.assertRaises(WorkerFinished):
                speaker._run()
        self.assertEqual(spoken, ['Warning, Stair ahead'])

    def test_pending_alerts_stay_bounded(self):
        speaker = Speaker.__new__(Speaker)
        speaker._queue = queue.Queue(maxsize=1)
        speaker.say('Warning, Stair ahead')
        speaker.say('Object detected ahead')
        self.assertEqual(speaker._queue.get_nowait(), 'Warning, Stair ahead')
        self.assertTrue(speaker._queue.empty())


if __name__ == '__main__':
    unittest.main()
