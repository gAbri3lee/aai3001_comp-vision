"""Keyboard shortcuts (decision U6) are ignored while a form control has focus."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'src' / 'static' / 'shortcuts.js'

CASES = [
    ({'key': ' ', 'target': {'tagName': 'BODY'}}, 'pause'),
    ({'key': 'm', 'target': {'tagName': 'DIV'}}, 'mute'),
    ({'key': 'M', 'target': {'tagName': 'BODY'}}, 'mute'),
    ({'key': 'c', 'target': {'tagName': 'BODY'}}, 'mode'),
    ({'key': 'x', 'target': {'tagName': 'BODY'}}, None),
    ({'key': 'm', 'target': {'tagName': 'INPUT'}}, None),       # typing or slider focused
    ({'key': ' ', 'target': {'tagName': 'input'}}, None),
    ({'key': 'c', 'target': {'tagName': 'TEXTAREA'}}, None),
    ({'key': 'c', 'target': {'tagName': 'SELECT'}}, None),
    ({'key': 'm', 'target': {'tagName': 'DIV', 'isContentEditable': True}}, None),
    ({'key': ' ', 'target': {'tagName': 'BUTTON'}}, None),      # the button handles Space itself
    ({'key': 'm', 'target': {'tagName': 'BUTTON'}}, 'mute'),
    ({'key': 'c', 'ctrlKey': True, 'target': {'tagName': 'BODY'}}, None),  # copy
    ({'key': 'm', 'metaKey': True, 'target': {'tagName': 'BODY'}}, None),
    ({'key': 'm', 'repeat': True, 'target': {'tagName': 'BODY'}}, None),
]


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
def test_shortcuts_are_ignored_in_form_controls():
    program = (f'const {{shortcutFor}} = require({json.dumps(str(SCRIPT))});'
               f'console.log(JSON.stringify({json.dumps([c for c, _ in CASES])}.map(shortcutFor)));')
    output = subprocess.run(['node', '-e', program], capture_output=True, text=True, check=True).stdout
    assert json.loads(output) == [expected for _, expected in CASES]


def test_page_loads_the_shortcuts_and_labels_the_buttons():
    from webapp import create_app
    from types import SimpleNamespace
    client = create_app(SimpleNamespace()).test_client()
    page = client.get('/').get_data(as_text=True)
    assert '/static/shortcuts.js' in page
    for label in ('<kbd>Space</kbd>', '<kbd>M</kbd>', '<kbd>C</kbd>'):
        assert label in page
    assert client.get('/static/shortcuts.js').status_code == 200
