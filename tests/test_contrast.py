"""Every colour pair on the page meets WCAG contrast (4.5 to 1 text, 3 to 1 large or non-text)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'check_contrast', Path(__file__).resolve().parents[1] / 'scripts' / 'check_contrast.py')
contrast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contrast)


def test_ratio_matches_known_values():
    assert round(contrast.ratio('#000000', '#ffffff'), 1) == 21.0
    assert round(contrast.ratio('#777777', '#ffffff'), 2) == 4.48


def test_all_page_colour_pairs_pass():
    failures = [row for row in contrast.check() if not row[-1]]
    assert failures == []


def test_page_uses_no_colour_outside_the_checked_tokens():
    """Colours in the stylesheet come from the :root tokens, so the check covers them."""
    page = contrast.PAGE.read_text()
    style = page[page.index('<style>'):page.index('</style>')]
    body = style[style.index('}', style.index(':root')):]
    assert '#' not in body.replace('#pause', '').replace('#mute', '')
    assert 'rgb' not in body
