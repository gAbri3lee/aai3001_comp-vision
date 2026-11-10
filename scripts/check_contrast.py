"""Check WCAG contrast of the page's colour pairs.

Reads the colour tokens from the :root block of src/templates/index.html and checks
every foreground and background pair the page uses. Normal text needs 4.5 to 1,
large text (the banner) and non-text indicators (focus outline, spoken border) 3 to 1.

    python scripts/check_contrast.py
"""
from pathlib import Path
import re
import sys

PAGE = Path(__file__).resolve().parents[1] / 'src' / 'templates' / 'index.html'

# (foreground token, background token, minimum ratio, where it is used)
PAIRS = [
    ('text', 'bg', 4.5, 'page text'),
    ('text', 'panel', 4.5, 'side panel text'),
    ('text', 'card', 4.5, 'cards, hazard rows'),
    ('text', 'button', 4.5, 'buttons'),
    ('muted', 'bg', 4.5, 'tagline, footer, legend'),
    ('muted', 'panel', 4.5, 'section headings, status line'),
    ('muted', 'card', 4.5, 'hazard details, far band'),
    ('muted', 'button', 4.5, 'secondary text on buttons'),
    ('on-accent', 'accent', 4.5, 'selected Local or Cloud button'),
    ('on-caution', 'caution-line', 4.5, 'pressed Pause or Mute'),
    ('warn-text', 'warn-bg', 4.5, 'Warning banner, problem messages'),
    ('caution-text', 'caution-bg', 4.5, 'Caution banner, warning messages'),
    ('note-text', 'note-bg', 4.5, 'Note banner, info messages'),
    ('warn-text', 'card', 4.5, 'near band label'),
    ('caution-text', 'card', 4.5, 'mid-close band label'),
    ('note-text', 'card', 4.5, 'mid band label'),
    ('spoken', 'card', 4.5, 'spoken tag'),
    ('overlay-text', 'overlay-bg', 4.5, 'video overlay, FPS badge'),
    ('muted', 'card', 3.0, 'empty banner text (large)'),
    ('focus', 'bg', 3.0, 'focus outline'),
    ('focus', 'panel', 3.0, 'focus outline in the side panel'),
    ('spoken', 'card', 3.0, 'spoken row border'),
    ('accent', 'bg', 3.0, 'mode badge border, analysis view border'),
    ('caution-line', 'bg', 3.0, 'cloud mode badge border'),
    ('text', 'card', 3.0, 'current legend entry border'),
]


def tokens(html):
    block = re.search(r':root\s*{(.*?)}', html, re.S).group(1)
    return {name: value.lower() for name, value in re.findall(r'--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;', block)}


def luminance(hex_colour):
    def channel(value):
        value /= 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def ratio(foreground, background):
    high, low = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def check(html=None):
    """Return [(fg, bg, ratio, minimum, use, ok)] for every pair."""
    colours = tokens(html if html is not None else PAGE.read_text())
    rows = []
    for fg, bg, minimum, use in PAIRS:
        value = ratio(colours[fg], colours[bg])
        rows.append((fg, bg, value, minimum, use, value >= minimum))
    return rows


def main():
    rows = check()
    print('| Foreground | Background | Ratio | Needed | Used for | Result |')
    print('|---|---|---:|---:|---|---|')
    for fg, bg, value, minimum, use, ok in rows:
        print(f'| {fg} | {bg} | {value:.1f} | {minimum} | {use} | {"pass" if ok else "FAIL"} |')
    failures = [row for row in rows if not row[-1]]
    print(f'\n{len(rows) - len(failures)} of {len(rows)} pairs pass')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
