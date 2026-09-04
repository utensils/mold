export type DesktopPlatform = "macos" | "linux" | "windows" | "unknown";

export function normalizePlatform(raw: string | undefined): DesktopPlatform {
  const platform = raw?.trim().toLowerCase();
  if (!platform) return "unknown";
  if (platform.includes("mac") || platform.includes("darwin")) return "macos";
  if (platform.includes("linux")) return "linux";
  if (platform.includes("win")) return "windows";
  return "unknown";
}

export function detectPlatform(
  tauriPlatform: string | undefined,
  browserPlatform: string | undefined,
  isTauri: boolean,
): DesktopPlatform {
  return normalizePlatform(tauriPlatform || (isTauri ? browserPlatform : undefined));
}

// Tauri exposes TAURI_ENV_PLATFORM for production builds, but the Vite server
// can start without it during `tauri dev`. WKWebView's platform keeps native
// macOS chrome (traffic-light insets and Command shortcuts) correct in dev.
export const CURRENT_PLATFORM = detectPlatform(
  import.meta.env.TAURI_ENV_PLATFORM,
  globalThis.navigator?.platform,
  "__TAURI_INTERNALS__" in globalThis,
);

export function applyPlatformAttribute(
  root: HTMLElement,
  platform: DesktopPlatform = CURRENT_PLATFORM,
) {
  root.dataset.platform = platform;
}

export function platformUi(raw: string | DesktopPlatform | undefined = CURRENT_PLATFORM) {
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

export const PLATFORM_UI = platformUi();

export function shortcutLabel(key: string): string {
  return `${PLATFORM_UI.modifierLabel}${key}`;
}

/** The platform's spelling of an Option/Alt chord, which carries no primary
 * modifier of its own. */
export function altShortcutLabel(key: string): string {
  return `${PLATFORM_UI.altLabel}${key}`;
}

/** The platform's spelling of a primary-modifier + Shift chord. */
export function shiftShortcutLabel(key: string): string {
  return `${PLATFORM_UI.modifierLabel}${PLATFORM_UI.shiftLabel}${key}`;
}

export function primaryModifierPressed(
  event: Pick<KeyboardEvent, "metaKey" | "ctrlKey" | "altKey">,
): boolean {
  if (event.altKey) return false;
  return PLATFORM_UI.isMacOS ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
}
