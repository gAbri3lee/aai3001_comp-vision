"""Priority legend from the config, and the bounded spoken log."""
from types import SimpleNamespace

from config import load_config
from helpers import FakeDetector, FakeSpeaker, running, wait_for
from webapp import SPOKEN_LOG_SIZE, create_app, priority_order


def test_priority_order_follows_the_config_severity():
    assert priority_order(load_config()['severity']) == [
        'Stair', 'Escalator', 'Wet Floor Sign', 'Pillar', 'People', 'Object']
    changed = {'Stair': .1, 'Escalator': .9, 'Wet Floor Sign': .5, 'Pillar': .7, 'People': .3}
    assert priority_order(changed) == ['Escalator', 'Pillar', 'Wet Floor Sign', 'People', 'Stair', 'Object']


def test_config_endpoint_serves_the_legend_and_the_page_does_not_hard_code_it():
    state = SimpleNamespace(settings={'severity': {'Stair': .2, 'Escalator': .9, 'Wet Floor Sign': .8,
                                                   'Pillar': .7, 'People': .5}})
    client = create_app(state).test_client()
    config = client.get('/config').json
    assert config['priority'] == ['Escalator', 'Wet Floor Sign', 'Pillar', 'People', 'Stair', 'Object']
    assert config['band_labels'] == {'near': 'Warning', 'mid-close': 'Caution', 'mid': 'Note'}
    assert (config['speech_rate_min'], config['speech_rate_max'], config['speech_rate_default']) == (120, 260, 180)
    page = client.get('/').get_data(as_text=True)
    legend = page[page.index('<ol class="legend"'):page.index('</ol>', page.index('<ol class="legend"'))]
    assert legend == '<ol class="legend" id="legend" aria-labelledby="legend-title">'  # filled from /config
    for text in ('level-chip', 'aria-live="assertive"', 'Spoken log', 'id="clear-log"'):
        assert text in page


def test_spoken_log_matches_the_speaker_and_is_bounded_and_clearable():
    speaker = FakeSpeaker()
    with running({'local': FakeDetector('yolov8n')}, speaker=speaker) as (state, client):
        assert wait_for(lambda: speaker.said == ['Warning, Stair ahead'])
        for _ in range(SPOKEN_LOG_SIZE + 3):
            client.post('/control', json={'action': 'test_voice'})
        client.post('/control', json={'action': 'replay_last'})
        log = client.get('/status').json['spoken_log']
        assert len(log) == SPOKEN_LOG_SIZE == 10
        # Newest first, and exactly what the speaker received.
        assert [entry['phrase'] for entry in log] == list(reversed(speaker.said))[:SPOKEN_LOG_SIZE]
        assert log[0]['kind'] == 'replay' and log[1]['kind'] == 'test'
        assert all(entry['backend'] == 'local' and isinstance(entry['time'], float) for entry in log)
        assert [e['time'] for e in log] == sorted((e['time'] for e in log), reverse=True)
        response = client.post('/control', json={'action': 'clear_log'})
        assert response.status_code == 200 and response.json['spoken_log'] == []
        client.post('/control', json={'muted': True})
        client.post('/control', json={'action': 'test_voice'})
        assert client.get('/status').json['spoken_log'] == []  # muted phrases are not spoken or logged


def test_clear_log_works_without_speech():
    with running({'local': FakeDetector('yolov8n')}) as (state, client):
        response = client.post('/control', json={'action': 'clear_log'})
        assert response.status_code == 200 and response.json['spoken_log'] == []
