/*
 * Platform conventions every Studio surface shares: what the primary modifier
 * is called, how a chord is spelled, and what this machine is called in copy.
 *
 * Everything here is PURE — the platform is always passed in, because each
 * shell learns it differently: the desktop app reads Tauri's build variable
 * (falling back to WKWebView's `navigator.platform` in dev), and the web SPA
 * reads the browser. Each shell owns that detection and its own bound
 * `CURRENT_PLATFORM`; only the conventions live here.
 */
export type Platform = "macos" | "linux" | "windows" | "unknown";

/** The desktop shell's historical name for the same union. */
export type DesktopPlatform = Platform;

export function normalizePlatform(raw: string | undefined): Platform {
  const platform = raw?.trim().toLowerCase();
  if (!platform) return "unknown";
  if (platform.includes("mac") || platform.includes("darwin")) return "macos";
  if (platform.includes("linux")) return "linux";
  if (platform.includes("win")) return "windows";
  return "unknown";
}

export function platformUi(raw: string | Platform | undefined = "unknown") {
  const platform = normalizePlatform(raw === "unknown" ? undefined : raw);
  const isMacOS = platform === "macos";
  return {
    isMacOS,
    modifier: isMacOS ? "Meta" : "Control",
    modifierLabel: isMacOS ? "⌘" : "Ctrl+",
    // Shift is a glyph inside the macOS chord and a named word inside the
    // Ctrl one, so it cannot be spelled by concatenating onto modifierLabel.
    shiftLabel: isMacOS ? "⇧" : "Shift+",
    // Option is a glyph on macOS and a named word everywhere else, for the
    // same reason Shift is.
    altLabel: isMacOS ? "⌥" : "Alt+",
    deviceLabel: isMacOS ? "This Mac" : "This device",
    // Each platform's own name for the app that opens a folder. "file manager"
    // is the honest generic on Linux, where there is no single one.
    fileManagerLabel: isMacOS
      ? "Finder"
      : platform === "windows"
        ? "File Explorer"
        : "file manager",
  } as const;
}

/**
 * Whether the event carries this platform's primary modifier — Command on
 * Apple platforms, Control everywhere else — and nothing that would make it a
 * different chord. On macOS Ctrl+E is move-to-end-of-line and Ctrl+↵ inserts a
 * line break, so a chord that accepted either modifier would take a key the
 * system already owns.
 */
export function primaryModifierPressed(
  event: Pick<KeyboardEvent, "metaKey" | "ctrlKey" | "altKey">,
  platform: string | Platform | undefined,
): boolean {
  if (event.altKey) return false;
  return platformUi(platform).isMacOS
    ? event.metaKey && !event.ctrlKey
    : event.ctrlKey && !event.metaKey;
}
