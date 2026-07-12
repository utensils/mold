/**
 * Grow a textarea vertically with its content, capped at `maxRows` (then it
 * scrolls internally). Deliberately JS instead of CSS `field-sizing`, which
 * WKWebView doesn't reliably support yet.
 */
export function autoGrowRows(el: HTMLTextAreaElement, maxRows = 8): void {
  const lineHeight = parseFloat(getComputedStyle(el).lineHeight) || 20;
  const maxHeight = Math.round(lineHeight * maxRows);
  // Collapse first so scrollHeight reports the true content height.
  el.style.height = "auto";
  const next = Math.min(el.scrollHeight, maxHeight);
  if (next > 0) el.style.height = `${next}px`;
  el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
}
