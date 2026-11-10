// Check the page layout in headless Chrome and save screenshots.
//
//   node scripts/check_layout.mjs http://127.0.0.1:5000 artifacts/screenshots [suffix]
//
// For each size it measures horizontal overflow (never allowed), page scrolling
// (not allowed at the two desktop sizes) and whether the main parts (header
// controls, video, banner, legend, status card) sit fully inside the viewport.
// Needs Google Chrome (set CHROME to override the path) and Node 22 or later.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const [base = "http://127.0.0.1:5000", outDir = "artifacts/screenshots", suffix = ""] = process.argv.slice(2);
const SIZES = [
  {name: "1920x1080", width: 1920, height: 1080, desktop: true, path: "/"},
  {name: "1366x768", width: 1366, height: 768, desktop: true, path: "/"},
  {name: "1366x768-debug", width: 1366, height: 768, desktop: true, path: "/?debug=1"},
  {name: "390x844", width: 390, height: 844, desktop: false, path: "/"},
];
const MAIN_PARTS = ["header", "#mode-local", "#mode-cloud", "#pause", "#mute", ".video-wrap",
                    "#alert-banner", ".legend-wrap", "#model-status"];
const PORT = 9333;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

const profile = mkdtempSync(join(tmpdir(), "visionaid-chrome-"));
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
                              "--no-first-run", "--no-default-browser-check", "about:blank"], {stdio: "ignore"});
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try {
    const pages = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    target = pages.find(p => p.type === "page");
  } catch {}
}
if (!target) { chrome.kill(); throw new Error("Chrome did not start"); }

const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
let nextId = 0;
const pending = new Map();
ws.onmessage = event => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    const {resolve, reject} = pending.get(message.id);
    pending.delete(message.id);
    message.error ? reject(new Error(message.error.message)) : resolve(message.result);
  }
};
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++nextId;
  pending.set(id, {resolve, reject});
  ws.send(JSON.stringify({id, method, params}));
});

const measure = `(() => {
  const root = document.documentElement;
  const parts = ${JSON.stringify(MAIN_PARTS)}.map(selector => {
    const el = document.querySelector(selector);
    if (!el) return {selector, missing: true};
    const r = el.getBoundingClientRect();
    return {selector, top: Math.round(r.top), bottom: Math.round(r.bottom), left: Math.round(r.left),
            right: Math.round(r.right),
            inside: r.top >= 0 && r.left >= 0 && r.bottom <= innerHeight + 0.5 && r.right <= innerWidth + 0.5};
  });
  return JSON.stringify({scrollWidth: root.scrollWidth, clientWidth: root.clientWidth,
                         scrollHeight: root.scrollHeight, innerHeight, innerWidth, parts,
                         banner: document.getElementById("alert-text").textContent,
                         fontSize: getComputedStyle(root).fontSize});
})()`;

mkdirSync(outDir, {recursive: true});
const results = [];
await send("Page.enable");
for (const size of SIZES) {
  await send("Emulation.setDeviceMetricsOverride", {width: size.width, height: size.height,
                                                    deviceScaleFactor: 1, mobile: !size.desktop});
  await send("Page.navigate", {url: base + size.path});
  await sleep(3500);  // status polls and video frames arrive
  const m = JSON.parse((await send("Runtime.evaluate", {expression: measure, returnByValue: true})).result.value);
  const shot = await send("Page.captureScreenshot", {format: "png"});
  const file = join(outDir, `gui2_${size.name}${suffix}.png`);
  writeFileSync(file, Buffer.from(shot.data, "base64"));
  const horizontal = m.scrollWidth > m.clientWidth;
  const pageScroll = m.scrollHeight > m.innerHeight;
  const outside = m.parts.filter(p => p.missing || !p.inside).map(p => p.selector);
  const ok = !horizontal && (!size.desktop || (!pageScroll && outside.length === 0));
  results.push({size: size.name, ok, horizontal_scroll: horizontal, page_scroll: pageScroll,
                main_parts_outside: size.desktop ? outside : "n/a (page scrolls on phones)",
                root_font: m.fontSize, banner: m.banner, screenshot: file});
}
ws.close();
chrome.kill();
console.log(JSON.stringify(results, null, 2));
process.exit(results.every(r => r.ok) ? 0 : 1);
