import { inTauri } from "./ipc";

/**
 * Open a URL in the user's default browser: the Tauri opener plugin inside
 * the app, `window.open` in the browser-dev fallback (same pattern as
 * RunPodView's console link and the App menu's API docs entry).
 */
export async function openExternal(url: string): Promise<void> {
  if (inTauri()) {
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    await openUrl(url);
  } else {
    window.open(url, "_blank", "noopener");
  }
}
