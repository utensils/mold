export type DesktopPlatform = "macos" | "linux" | "windows" | "unknown";

export function normalizePlatform(raw: string | undefined): DesktopPlatform {
  switch (raw?.toLowerCase()) {
    case "darwin":
    case "macos":
      return "macos";
    case "linux":
      return "linux";
    case "win32":
    case "windows":
      return "windows";
    default:
      return "unknown";
  }
}

export const CURRENT_PLATFORM = normalizePlatform(import.meta.env.TAURI_ENV_PLATFORM);

export function platformUi(raw: string | DesktopPlatform | undefined = CURRENT_PLATFORM) {
  const platform = normalizePlatform(raw === "unknown" ? undefined : raw);
  const isMacOS = platform === "macos";
  return {
    isMacOS,
    modifier: isMacOS ? "Meta" : "Control",
    modifierLabel: isMacOS ? "⌘" : "Ctrl+",
    deviceLabel: isMacOS ? "This Mac" : "This device",
    fileManagerLabel: isMacOS ? "Finder" : "file manager",
  } as const;
}

export const PLATFORM_UI = platformUi();

export function shortcutLabel(key: string): string {
  return `${PLATFORM_UI.modifierLabel}${key}`;
}

export function primaryModifierPressed(event: Pick<KeyboardEvent, "metaKey" | "ctrlKey">): boolean {
  return PLATFORM_UI.isMacOS ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
}
