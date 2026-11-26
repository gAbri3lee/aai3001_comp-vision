"""Browser end-to-end check of the video upload on a deployed (or local) VisionAid Vercel page.

    pip install playwright            # uses an installed Edge/Chrome: no browser download needed
    python deploy/vercel/app.py       # in another terminal, or use the live URL
    python deploy/vercel/check_video_ui.py http://127.0.0.1:3000
    python deploy/vercel/check_video_ui.py https://<your-app>.vercel.app --bypass-secret <protection bypass secret>

It drives a real browser: uploads a photo and MP4 videos, presses Play / Pause / Restart, replaces the video,
resizes the window, and slows or breaks the detector with request interception, checking that
  * only small JPEG frames reach POST /api/detect (never the video),
  * never more than one request is in flight, and slow replies make the page skip frames,
  * boxes are drawn where the detector said (including a letterboxed portrait video, before and after a resize),
  * late replies from a previous video or position are never drawn,
  * a failing detector shows an error while the video keeps playing, and recovers,
  * the page raises no JavaScript errors.
Fault-injection cases (slow or failing detector) are answered locally by the test; every other case talks to the real
endpoint, so they only verify cloud detection when BASE_URL is the deployed site.
Exit status is 0 only if every check passes.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile

try:
    from playwright.async_api import async_playwright
except ImportError:
    sys.exit('Install Playwright first: pip install playwright (an installed Edge or Chrome is used)')

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / 'demo' / 'videos'
LANDSCAPE = VIDEOS / 'Walk to the subway stairs.mp4'          # 1280x720
PORTRAIT = VIDEOS / 'Wet floor sign - field sequence.mp4'     # 204x360: letterboxed by the player
LONG = VIDEOS / 'People passing through a shared space.mp4'         # 768x432, 50 s: outlasts the fault-injection waits
SQUARE = VIDEOS / 'Escalator - field sequence.mp4'            # 640x640
PHOTO = sorted((ROOT / 'golden' / 'images').glob('*.jpg'))[0]
RATE_NOTE = 'Checks per second'

results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok), detail))
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'  [{detail}]' if detail else ''), flush=True)
    return ok


class Probe:
    """Watches one page: JS errors and every /api/detect request."""

    def __init__(self, page):
        self.page, self.js_errors, self.requests, self.responses = page, [], [], []
        self.inflight = self.max_inflight = 0
        self._entries = {}
        self.expect_http_errors = False   # set while a case injects 4xx/5xx replies or network failures
        self.other_posts = []
        page.on('pageerror', lambda e: self.js_errors.append(str(e)))
        page.on('console', self._console)
        page.on('request', self._request)
        page.on('requestfinished', lambda r: self._done(r))
        page.on('requestfailed', lambda r: self._done(r))
        page.on('response', lambda r: asyncio.ensure_future(self._response(r)))

    def _console(self, msg):
        if msg.type == 'error' and not (self.expect_http_errors and 'Failed to load resource' in msg.text):
            self.js_errors.append('console: ' + msg.text)

    def _request(self, req):
        if req.url.split('?')[0].endswith('/api/detect'):
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            # Playwright cannot read Blob request bodies, so the size comes from Content-Length and "is a JPEG" from the
            # Content-Type plus the server accepting it (app.py rejects anything Pillow does not decode as JPEG with 400).
            entry = {'method': req.method, 'type': req.headers.get('content-type'), 'size': 0,
                     'jpeg': req.headers.get('content-type') == 'image/jpeg', 'status': None}
            self.requests.append(entry)
            self._entries[id(req)] = entry
            asyncio.ensure_future(self._size(req, entry))
        elif req.method != 'GET':
            self.other_posts.append(req.url)

    async def _size(self, req, entry):
        try:
            entry['size'] = int((await req.all_headers()).get('content-length', 0))
        except Exception:
            pass

    def _done(self, req):
        if req.url.split('?')[0].endswith('/api/detect'):
            self.inflight = max(0, self.inflight - 1)

    async def _response(self, resp):
        entry = self._entries.get(id(resp.request))
        if entry:
            entry['status'] = resp.status
        if resp.url.split('?')[0].endswith('/api/detect') and resp.status == 200:
            try:
                self.responses.append(await resp.json())
            except Exception:
                pass

    def reset(self):
        self.requests, self.responses, self.max_inflight = [], [], self.inflight


async def text(page, sel):
    return (await page.locator(sel).text_content()) or ''


async def settle(probe, quiet=1.2, limit=15):
    """Wait until no request is in flight and none has started for `quiet` seconds."""
    last, stable = -1, 0.0
    for _ in range(int(limit / 0.1)):
        await asyncio.sleep(0.1)
        n = len(probe.requests)
        stable = stable + 0.1 if (n == last and probe.inflight == 0) else 0.0
        last = n
        if stable >= quiet:
            return True
    return False


async def current_time(page):
    return await page.evaluate("document.getElementById('video').currentTime")


async def overlay_state(page):
    """Opaque-pixel bounding box of the overlay and the geometry of the picture inside the video element (CSS px)."""
    return await page.evaluate("""() => {
      const v = document.getElementById('video'), o = document.getElementById('overlay');
      const r = o.getBoundingClientRect(), vr = v.getBoundingClientRect();
      const dpr = o.width / Math.max(1, r.width), ctx = o.getContext('2d');
      const px = ctx.getImageData(0, 0, o.width, o.height).data;
      let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1, count = 0;
      for (let y = 0; y < o.height; y++) for (let x = 0; x < o.width; x++) {
        if (px[(y * o.width + x) * 4 + 3] > 0) { count++; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
      }
      const out = { count, bbox: count ? [x0 / dpr, y0 / dpr, (x1 + 1) / dpr, (y1 + 1) / dpr] : null,
                    cw: v.clientWidth, ch: v.clientHeight, vw: v.videoWidth, vh: v.videoHeight,
                    overlayRect: [r.left - vr.left, r.top - vr.top, r.width, r.height] };
      return out;
    }""")


def expected_boxes(detections, size, state):
    """Boxes in overlay CSS pixels from the detector's pixels, mirroring object-fit: contain."""
    cw, ch, aspect = state['cw'], state['ch'], state['vw'] / state['vh']
    rw, rh = cw, cw / aspect
    if rh > ch:
        rh, rw = ch, ch * aspect
    ox, oy, sx, sy = (cw - rw) / 2, (ch - rh) / 2, rw / size[0], rh / size[1]
    return [(ox + b[0] * sx, oy + b[1] * sy, ox + b[2] * sx, oy + b[3] * sy) for b in (d['box_xyxy'] for d in detections)], (ox, oy, rw, rh)


async def edges_drawn(page, boxes):
    """Does the overlay have ink at the midpoint of every box edge?"""
    return await page.evaluate("""(boxes) => {
      const o = document.getElementById('overlay'), dpr = o.width / o.clientWidth, ctx = o.getContext('2d');
      const px = ctx.getImageData(0, 0, o.width, o.height).data;
      const ink = (cx, cy) => { const x = Math.round(cx * dpr), y = Math.round(cy * dpr);
        for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx >= 0 && yy >= 0 && xx < o.width && yy < o.height && px[(yy * o.width + xx) * 4 + 3] > 0) return true; }
        return false; };
      return boxes.map(([x1, y1, x2, y2]) => [ink((x1 + x2) / 2, y1), ink((x1 + x2) / 2, y2), ink(x1, (y1 + y2) / 2), ink(x2, (y1 + y2) / 2)]);
    }""", boxes)


async def check_alignment(page, probe, label):
    """Compare what is drawn with the detector's last reply for the displayed frame (video must be paused and settled)."""
    state = await overlay_state(page)
    if not probe.responses:
        return check(f'{label}: a detector reply is available', False)
    reply = probe.responses[-1]
    dets = reply['detections']
    boxes, (ox, oy, rw, rh) = expected_boxes(dets, reply['image_size'], state)
    ok_size = abs(state['overlayRect'][2] - state['cw']) < 1.5 and abs(state['overlayRect'][3] - state['ch']) < 1.5 \
        and abs(state['overlayRect'][0]) < 1.5 and abs(state['overlayRect'][1]) < 1.5
    check(f'{label}: overlay covers the video element exactly', ok_size, f"overlay={state['overlayRect']} video={state['cw']}x{state['ch']}")
    if not dets:
        return check(f'{label}: no boxes drawn when the detector found nothing', state['count'] == 0, f"ink pixels={state['count']}")
    edges = await edges_drawn(page, [list(b) for b in boxes])
    check(f'{label}: ink on all four edges of all {len(boxes)} boxes', all(all(e) for e in edges), json.dumps(edges))
    xs1, ys1, xs2, ys2 = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    bb = state['bbox']
    tol = 6                       # line width; labels may stick out above/right of the union of the boxes
    inside = bb and bb[0] >= xs1 - tol and bb[1] >= ys1 - 24 - tol and bb[2] <= xs2 + 260 and bb[3] <= ys2 + tol
    check(f'{label}: nothing is drawn away from the boxes', inside, f'ink bbox={bb and [round(v) for v in bb]} boxes=[{xs1:.0f},{ys1:.0f},{xs2:.0f},{ys2:.0f}]')
    inside_picture = xs1 >= ox - tol and ys1 >= oy - tol and xs2 <= ox + rw + tol and ys2 <= oy + rh + tol
    return check(f'{label}: boxes lie inside the picture, not the black bars', inside_picture, f'picture=({ox:.0f},{oy:.0f},{rw:.0f},{rh:.0f})')


async def wait_for(page, probe, predicate_js, timeout=20.0, arg=None):
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        if await page.evaluate(predicate_js, arg):
            return True
        await asyncio.sleep(0.15)
    return False


HAS_INK = "() => { const o = document.getElementById('overlay'); return o.width > 0 && o.getContext('2d').getImageData(0, 0, o.width, o.height).data.some((v, i) => i % 4 === 3 && v > 0); }"


async def open_page(browser, base, extra_headers, viewport=(1100, 900)):
    context = await browser.new_context(viewport={'width': viewport[0], 'height': viewport[1]}, extra_http_headers=extra_headers)
    page = await context.new_page()
    probe = Probe(page)
    await page.goto(base + '/', wait_until='load')
    return context, page, probe


async def load_video(page, path):
    await page.set_input_files('#file', str(path))
    await page.wait_for_function("() => document.getElementById('video').readyState >= 2 || /cannot play|could not be read|no video/i.test(document.getElementById('status').textContent)", timeout=15000)


async def case_photo(browser, base, headers):
    context, page, probe = await open_page(browser, base, headers)
    await page.set_input_files('#file', str(PHOTO))
    await page.wait_for_function("() => document.getElementById('stats').textContent.includes('Model:')", timeout=60000)
    check('photo: still analysed (alert, list and stats shown)', len(await text(page, '#alert')) > 5 and await page.locator('#stats span').count() == 4)
    check('photo: canvas visible, video panel hidden', await page.locator('#canvas').is_visible() and not await page.locator('#videoPanel').is_visible())
    check('photo: sent as one JPEG request, accepted by the server', len(probe.requests) == 1 and probe.requests[0]['jpeg'] and probe.requests[0]['status'] == 200)
    await page.set_input_files('#file', str(SQUARE))                    # photo -> video switch
    await page.wait_for_function("() => document.getElementById('video').readyState >= 2", timeout=15000)
    check('photo -> video: photo canvas hidden, video player shown', not await page.locator('#canvas').is_visible() and await page.locator('#video').is_visible())
    await page.set_input_files('#file', str(PHOTO))                                           # video -> photo switch
    await page.wait_for_function("() => document.getElementById('stats').textContent.includes('Model:') && !document.getElementById('canvas').hidden", timeout=60000)
    check('video -> photo: photo works again, video panel hidden', await page.locator('#canvas').is_visible() and not await page.locator('#videoPanel').is_visible())
    check('photo: no JavaScript errors', not probe.js_errors, '; '.join(probe.js_errors))
    await context.close()


async def case_playback(browser, base, headers):
    context, page, probe = await open_page(browser, base, headers)
    video_size = LANDSCAPE.stat().st_size
    await load_video(page, LANDSCAPE)
    check('video: loads and shows Ready', 'Ready' in await text(page, '#status'), await text(page, '#status'))
    check('video: player visible with Play enabled, Pause disabled',
          await page.locator('#video').is_visible() and await page.locator('#play').is_enabled() and not await page.locator('#pause').is_enabled())
    check('video: nothing sent before Play', len(probe.requests) == 0)

    await page.click('#play')
    await asyncio.sleep(0.4)
    check('play: Pause enabled, Play disabled', await page.locator('#pause').is_enabled() and not await page.locator('#play').is_enabled())
    await asyncio.sleep(6.0)
    t = await current_time(page)
    check('play: video advances while detecting', t > 4.5, f't={t:.1f}s')
    n = len(probe.requests)
    check('play: about 2 requests per second (not more)', 6 <= n <= 14, f'{n} requests in ~6.4 s')
    done = [r for r in probe.requests if r['status'] is not None]
    check('play: every request is a small JPEG POST that the server decoded (HTTP 200)',
          done and all(r['method'] == 'POST' and r['jpeg'] and r['status'] == 200 and 0 < r['size'] < 400_000 for r in done),
          f"{len(done)} replies, max frame {max((r['size'] for r in probe.requests), default=0) // 1024} KB")
    check('play: never more than one request in flight', probe.max_inflight <= 1, f'max in flight={probe.max_inflight}')
    check('play: the video file is not uploaded', sum(r['size'] for r in probe.requests) < video_size and not probe.other_posts,
          f"uploaded {sum(r['size'] for r in probe.requests) // 1024} KB in total vs a {video_size // 1024} KB video")
    check('play: summary and stats update', 'Requests' in await text(page, '#stats'))
    got_boxes = await wait_for(page, probe, HAS_INK, 15)
    check('play: boxes appear over the video', got_boxes)

    await page.click('#pause')
    await settle(probe)
    paused_at, count = await current_time(page), len(probe.requests)
    await asyncio.sleep(2.0)
    check('pause: video stops and requests stop', abs(await current_time(page) - paused_at) < 0.05 and len(probe.requests) == count, f'{len(probe.requests) - count} extra requests')
    check('pause: Play enabled, Pause disabled', await page.locator('#play').is_enabled() and not await page.locator('#pause').is_enabled())

    # Alignment on a landscape video, then after the window is resized.
    ink = await page.evaluate(HAS_INK)
    if not ink:        # nothing detected on this frame: step forward until the detector finds something
        await page.click('#play')
        await wait_for(page, probe, HAS_INK, 20)
        await page.click('#pause')
        await settle(probe)
    await check_alignment(page, probe, 'landscape')
    await page.set_viewport_size({'width': 560, 'height': 800})
    await asyncio.sleep(0.6)
    await check_alignment(page, probe, 'landscape after resize to 560px')
    await page.set_viewport_size({'width': 1300, 'height': 700})
    await asyncio.sleep(0.6)
    await check_alignment(page, probe, 'landscape after resize to 1300x700')

    # Resume.
    before = len(probe.requests)
    t_before = await current_time(page)
    await page.click('#play')
    await asyncio.sleep(3.0)
    check('resume: playback continues from where it paused and detection resumes',
          await current_time(page) > t_before + 2 and len(probe.requests) > before + 2, f'+{len(probe.requests) - before} requests')

    # Restart.
    await page.click('#restart')
    await asyncio.sleep(0.15)
    check('restart: jumps to the start and plays', await current_time(page) < 1.5 and await page.locator('#pause').is_enabled(), f't={await current_time(page):.2f}s')
    check('restart: old boxes removed immediately', not await page.evaluate(HAS_INK) or await current_time(page) < 1.5)
    await asyncio.sleep(3.0)
    check('restart: detection continues afterwards', len(probe.requests) > before + 4)

    # Play to the end (sped up so the test is short).
    await page.evaluate("document.getElementById('video').playbackRate = 8")
    ended = await wait_for(page, probe, "() => document.getElementById('video').ended", 30)
    check('end: video reaches the end', ended)
    await settle(probe, quiet=1.5)
    count = len(probe.requests)
    await asyncio.sleep(2.5)
    check('end: requests stop and the page says so', len(probe.requests) == count and 'finished' in await text(page, '#status'), await text(page, '#status'))
    check('end: Play re-enabled', await page.locator('#play').is_enabled())
    await page.evaluate("document.getElementById('video').playbackRate = 1")
    await page.click('#play')
    await asyncio.sleep(2.0)
    check('end: Play after the end replays from the start', await current_time(page) < 4 and await page.locator('#pause').is_enabled(), f't={await current_time(page):.1f}s')
    await page.click('#pause')
    check('playback: no JavaScript errors', not probe.js_errors, '; '.join(probe.js_errors))
    await context.close()


async def case_portrait_alignment(browser, base, headers):
    context, page, probe = await open_page(browser, base, headers)
    await load_video(page, PORTRAIT)
    await page.click('#play')
    found = await wait_for(page, probe, HAS_INK, 25)
    await page.click('#pause')
    await settle(probe)
    check('portrait: boxes appear', found)
    if found:
        state = await overlay_state(page)
        check('portrait: the player letterboxes it (so the mapping is really tested)', state['cw'] / state['ch'] > state['vw'] / state['vh'] + 0.2, f"{state['cw']}x{state['ch']} for {state['vw']}x{state['vh']}")
        await check_alignment(page, probe, 'portrait (pillarboxed)')
        await page.set_viewport_size({'width': 700, 'height': 700})
        await asyncio.sleep(0.6)
        await check_alignment(page, probe, 'portrait after resize')
    check('portrait: no JavaScript errors', not probe.js_errors, '; '.join(probe.js_errors))
    await context.close()


async def case_slow_server(browser, base, headers):
    context, page, probe = await open_page(browser, base, headers)

    async def slow(route):
        await asyncio.sleep(2.5)
        await route.continue_()
    await page.route('**/api/detect', slow)
    await load_video(page, LANDSCAPE)
    await page.click('#play')
    await asyncio.sleep(10.0)
    t = await current_time(page)
    n = len(probe.requests)
    check('slow server (2.5 s replies): playback stays responsive', t > 8.0, f't={t:.1f}s after 10 s')
    check('slow server: at most one request in flight', probe.max_inflight <= 1, f'max in flight={probe.max_inflight}')
    check('slow server: requests do not accumulate (about one per 2.5 s, not two per second)', n <= 5, f'{n} requests in 10 s (an unthrottled page would send ~20)')
    await page.wait_for_function("() => /Skipped while busy [1-9]/.test(document.getElementById('stats').textContent)", timeout=10000)
    check('slow server: skipped frames are counted', True, await text(page, '#stats'))
    await page.click('#pause')
    check('slow server: no JavaScript errors', not probe.js_errors, '; '.join(probe.js_errors))
    await context.close()


async def case_stale_replies(browser, base, headers):
    """A reply that arrives after a restart or after the file was switched must never be drawn."""
    fake = {'detections': [{'class_id': 2, 'class': 'Stair', 'confidence': 0.99, 'box_xyxy': [10, 10, 600, 300]}],
            'latency_ms': 1, 'model': 'fake-stale', 'image_size': [640, 360]}

    async def scenario(action, label):
        context, page, probe = await open_page(browser, base, headers)
        probe.expect_http_errors = True
        state = {'n': 0, 'held': asyncio.Event(), 'release': asyncio.Event()}

        async def route_handler(route):
            state['n'] += 1
            if state['n'] > 1 and not state.get('hold_all'):
                await route.continue_()                        # later requests reach the real detector
                return
            state['held'].set()
            await state['release'].wait()                      # the first reply is held back, then delivered late
            try:
                await route.fulfill(status=200, content_type='application/json', body=json.dumps(fake))
            except Exception:
                pass                                           # the page aborted the request: also fine
        await page.route('**/api/detect', route_handler)
        await load_video(page, LANDSCAPE)
        await page.click('#play')
        await asyncio.wait_for(state['held'].wait(), 10)
        await action(page)                                     # restart, or switch to another video
        state['release'].set()                                 # now deliver the stale reply
        shown = False
        for _ in range(25):                                    # watch for 2.5 s
            await asyncio.sleep(0.1)
            shown = shown or 'fake-stale' in await text(page, '#stats') or 'Stair' in await text(page, '#alert')
        check(f'stale ({label}): the late reply is never drawn or summarised', not shown)
        return context, page, probe

    async def restart(page):
        await page.click('#restart')

    async def switch(page):
        await load_video(page, SQUARE)

    context, page, probe = await scenario(restart, 'after Restart')
    await page.click('#pause')
    check('stale (after Restart): later real replies are accepted again', await wait_for(page, probe, "() => document.getElementById('stats').textContent.includes('yolov8n-onnx')", 15))
    check('stale (after Restart): no JavaScript errors', not probe.js_errors, '; '.join(probe.js_errors))
    await context.close()

    context, page, probe = await scenario(switch, 'after switching video')
    check('stale (after switch): new video starts idle with nothing drawn', not await page.evaluate(HAS_INK) and await text(page, '#alert') == '' and await page.locator('#list li').count() == 0)
    n = len(probe.requests)
    await asyncio.sleep(1.5)
    check('stale (after switch): no frames are sent for the new video until Play', len(probe.requests) == n)
    check('stale (after switch): no JavaScript errors', not probe.js_errors, '; '.join(probe.js_errors))
    await context.close()


async def case_backend_failure(browser, base, headers):
    context, page, probe = await open_page(browser, base, headers)
    probe.expect_http_errors = True
    mode = {'v': 'down'}

    async def flaky(route):
        if mode['v'] == 'down':
            await route.abort('connectionrefused')
        elif mode['v'] == '503':
            await route.fulfill(status=503, content_type='application/json', body='{"error":"Detector inference failed"}')
        else:
            await route.continue_()
    await page.route('**/api/detect', flaky)
    await load_video(page, LONG)
    await page.click('#play')
    await asyncio.sleep(4.0)
    status = await text(page, '#status')
    check('backend unreachable: clear error, video keeps playing', 'Could not reach the detector' in status and await current_time(page) > 3.0, f'{status!r} t={await current_time(page):.1f}s')
    n = len(probe.requests)
    await asyncio.sleep(4.0)
    check('backend unreachable: retries back off instead of hammering', len(probe.requests) - n <= 3, f'{len(probe.requests) - n} retries in 4 s')
    mode['v'] = '503'
    await asyncio.sleep(9.0)
    status = await text(page, '#status')
    check('backend 503: server message shown, still playing', 'Detector inference failed' in status and await current_time(page) > 10, status)
    check('backend failure: controls still work', await page.locator('#pause').is_enabled())
    mode['v'] = 'ok'
    recovered = await wait_for(page, probe, "() => document.getElementById('stats').textContent.includes('Model:') && !document.getElementById('status').classList.contains('error')", 25)
    check('backend recovers: detection resumes and the error clears', recovered, await text(page, '#status'))
    await page.click('#pause')
    check('backend failure: no JavaScript errors', not probe.js_errors, '; '.join(probe.js_errors))
    await context.close()


async def case_bad_files(browser, base, headers):
    context, page, probe = await open_page(browser, base, headers)
    tmp = Path(tempfile.mkdtemp(prefix='visionaid_ui_'))
    (tmp / 'notes.txt').write_text('not media')
    (tmp / 'broken.mp4').write_bytes(b'this is not a real mp4 file' * 100)
    (tmp / 'broken.webm').write_bytes(b'\x1a\x45\xdf\xa3' + b'garbage' * 200)
    await page.set_input_files('#file', str(tmp / 'notes.txt'))
    await asyncio.sleep(0.3)
    check('invalid file type: clear message', 'not a photo or an MP4/WebM video' in await text(page, '#status'), await text(page, '#status'))
    for name in ('broken.mp4', 'broken.webm'):
        await page.set_input_files('#file', str(tmp / name))
        ok = await wait_for(page, probe, "() => /cannot play|could not be read|no video/i.test(document.getElementById('status').textContent)", 10)
        check(f'{name}: unreadable video gives a clear error', ok, await text(page, '#status'))
        check(f'{name}: player hidden, nothing sent', not await page.locator('#video').is_visible() and not probe.requests)
    await load_video(page, SQUARE)
    check('after errors: a good video loads normally', 'Ready' in await text(page, '#status'))
    await page.click('#play')
    await asyncio.sleep(2.5)
    check('after errors: and its frames are detected', len(probe.requests) >= 2 and probe.requests[0]['jpeg'])
    await page.click('#pause')
    check('bad files: no JavaScript errors', not probe.js_errors, '; '.join(probe.js_errors))
    await context.close()


async def case_webm(browser, base, headers, webm):
    context, page, probe = await open_page(browser, base, headers)
    await load_video(page, webm)
    check('webm: loads', 'Ready' in await text(page, '#status'), await text(page, '#status'))
    await page.click('#play')
    await asyncio.sleep(3.0)
    check('webm: frames reach /api/detect as JPEG', len(probe.requests) >= 2 and all(r['jpeg'] for r in probe.requests))
    await page.click('#pause')
    check('webm: no JavaScript errors', not probe.js_errors, '; '.join(probe.js_errors))
    await context.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('base_url')
    parser.add_argument('--browser', default='msedge', help='Playwright channel: msedge, chrome, or "chromium" for its own download')
    parser.add_argument('--bypass-secret', help='Vercel Deployment Protection bypass secret, if the site requires sign-in')
    parser.add_argument('--webm', type=Path, help='optional WebM file to test as well')
    parser.add_argument('--skip', nargs='*', default=[], help='case names to skip, e.g. slow_server')
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    headers = {'x-vercel-protection-bypass': args.bypass_secret} if args.bypass_secret else {}
    cases = [('photo', case_photo), ('playback', case_playback), ('portrait', case_portrait_alignment),
             ('slow_server', case_slow_server), ('stale', case_stale_replies), ('backend_failure', case_backend_failure),
             ('bad_files', case_bad_files)]
    if args.webm:
        cases.append(('webm', lambda b, u, h: case_webm(b, u, h, args.webm)))
    async with async_playwright() as p:
        kwargs = {} if args.browser == 'chromium' else {'channel': args.browser}
        browser = await p.chromium.launch(headless=True, args=['--autoplay-policy=no-user-gesture-required'], **kwargs)
        for name, fn in cases:
            if name in args.skip:
                continue
            print(f'--- {name}', flush=True)
            try:
                await fn(browser, base, headers)
            except Exception as exc:      # a crashed case is a failure, not a skipped one
                check(f'{name}: ran to completion', False, f'{type(exc).__name__}: {exc}')
        await browser.close()
    failed = [r for r in results if not r[1]]
    print(f'\n{len(results) - len(failed)}/{len(results)} checks passed')
    for name, _, detail in failed:
        print('FAILED:', name, detail)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    asyncio.run(main())
