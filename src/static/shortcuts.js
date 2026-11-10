// Keyboard shortcuts (decision U6): Space pauses or resumes, M mutes or unmutes,
// C toggles Local or Cloud. A pure function so it can be tested without a browser.
const SHORTCUTS = {" ": "pause", "m": "mute", "c": "mode"};
const FORM_TAGS = ["INPUT", "TEXTAREA", "SELECT"];

// Returns "pause", "mute", "mode" or null for a keydown event.
function shortcutFor(event) {
  if (!event || event.ctrlKey || event.metaKey || event.altKey || event.repeat) return null;
  const target = event.target || {};
  const tag = String(target.tagName || "").toUpperCase();
  // Never while typing in or adjusting a form control.
  if (FORM_TAGS.includes(tag) || target.isContentEditable) return null;
  const key = String(event.key || "").toLowerCase();
  // Space on a focused button or summary already presses it; do not act twice.
  if (key === " " && (tag === "BUTTON" || tag === "SUMMARY")) return null;
  return SHORTCUTS[key] || null;
}

if (typeof module !== "undefined") module.exports = {shortcutFor, SHORTCUTS};
