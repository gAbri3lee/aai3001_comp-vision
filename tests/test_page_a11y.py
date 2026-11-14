"""Page markup: labelled controls, live banner, no colour-only state, no dashes."""
from html.parser import HTMLParser
from types import SimpleNamespace

from webapp import create_app


class Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements, self.label_for, self.stack, self.text = [], set(), [], {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.elements.append((tag, attrs, [t for t, _ in self.stack]))
        if tag == 'label' and 'for' in attrs:
            self.label_for.add(attrs['for'])
        if tag not in ('input', 'meta', 'link', 'br', 'img'):
            self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        while self.stack:
            if self.stack.pop()[0] == tag:
                break

    def handle_data(self, data):
        for tag, attrs in self.stack:
            if 'id' in attrs:
                self.text[attrs['id']] = self.text.get(attrs['id'], '') + data


def page():
    html = create_app(SimpleNamespace()).test_client().get('/').get_data(as_text=True)
    collector = Collector()
    collector.feed(html)
    return html, collector


def test_every_control_has_a_label_and_is_a_real_control():
    html, c = page()
    for tag, attrs, parents in c.elements:
        if tag in ('input', 'select', 'textarea'):
            assert attrs.get('id') in c.label_for or 'label' in parents, f'{tag} {attrs} has no label'
        if tag == 'button':
            assert attrs.get('type') == 'button'
            assert c.text.get(attrs['id'], '').strip(), f'button {attrs["id"]} has no text'
    assert 'onclick' not in html and '<div role="button"' not in html


def test_banner_is_an_assertive_live_region_and_states_are_written_out():
    html, c = page()
    banner = next(attrs for tag, attrs, _ in c.elements if attrs.get('id') == 'alert-banner')
    assert banner['aria-live'] == 'assertive' and banner['aria-atomic'] == 'true'
    for toggle in ('pause', 'mute', 'mode-local', 'mode-cloud'):
        attrs = next(a for t, a, _ in c.elements if a.get('id') == toggle)
        assert attrs['aria-pressed'] in ('true', 'false')
    assert ':focus-visible' in html and 'outline: 3px solid var(--focus)' in html
    # Levels and messages carry words, not only colour.
    for text in ('level-chip', 'content: "Problem:"', 'content: "Warning:"', 'content: "Note:"',
                 '"spoken"', 'selected (muted)', 'aria-current'):
        assert text in html


def test_no_em_or_en_dashes_in_the_page_or_scripts():
    from pathlib import Path
    html, _ = page()
    shortcuts = (Path(__file__).resolve().parents[1] / 'src' / 'static' / 'shortcuts.js').read_text()
    for text in (html, shortcuts):
        assert '\u2014' not in text and '\u2013' not in text
