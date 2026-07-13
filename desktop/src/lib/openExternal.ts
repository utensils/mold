import { inTauri } from "./ipc";

/**
 * Open a URL in the user's default browser: the Tauri opener plugin inside
 * the app, `window.open` in the browser-dev fallback (same pattern as
 * RunPodView's console link and the App menu's API docs entry).
 *
 * Only `http:` / `https:` URLs are opened. Callers feed this
 * server-supplied strings (e.g. a catalog row's `page_url` from whatever
 * `mold serve` host the app is connected to — including one-click mDNS
 * connects), and the OS opener would happily launch `file:` or custom
 * app schemes. Anything else is a silent no-op.
 */
export async function openExternal(url: string): Promise<void> {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return;
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return;

  if (inTauri()) {
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    await openUrl(url);
  } else {
    window.open(url, "_blank", "noopener,noreferrer");
  }
}
