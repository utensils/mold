/**
 * Read a `File` (drag-drop or <input type=file>) to base64 with no data-URI
 * prefix — the shape mold-core expects for `source_image` / `mask_image` /
 * `control_image` on the wire. Works in WKWebView and a plain browser.
 *
 * TODO(M6): use the native Tauri file dialog + fs plugin when picking on macOS
 * so large images don't round-trip through the WebView.
 */
export function fileToBase64(file: File): Promise<string> {
  return blobToBase64(file);
}

/** Read any `Blob` (e.g. a `fetch().blob()` of an authed gallery image) to
 * base64 with no data-URI prefix — the shape mold-core expects on the wire. */
export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result);
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the file"));
    reader.readAsDataURL(blob);
  });
}

/** Object URL for a base64 payload so a `<img>` can preview it. */
export function base64ToDataUrl(b64: string, mime = "image/png"): string {
  return `data:${mime};base64,${b64}`;
}
