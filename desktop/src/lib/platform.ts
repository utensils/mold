/*
 * The desktop shell's platform binding. The conventions themselves (the
 * modifier, its spellings, this machine's name) are shared with the web SPA in
 * `@studio/lib/platform`; what lives here is how the DESKTOP learns which
 * platform it is running on, and everything bound to that answer.
 */
import {
  normalizePlatform,
  platformUi as platformUiFor,
  primaryModifierPressed as primaryModifierPressedOn,
  type Platform,
} from "@studio/lib/platform";

export { normalizePlatform };

/** The shared conventions, defaulting to the platform THIS shell detected —
 * a bare `platformUi()` must never quietly mean "unknown" on a Mac. */
export function platformUi(raw: string | Platform | undefined = CURRENT_PLATFORM) {
  return platformUiFor(raw);
}
export type { DesktopPlatform } from "@studio/lib/platform";

export function detectPlatform(
  tauriPlatform: string | undefined,
  browserPlatform: string | undefined,
  isTauri: boolean,
): Platform {
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

export function applyPlatformAttribute(root: HTMLElement, platform: Platform = CURRENT_PLATFORM) {
  root.dataset.platform = platform;
}

export const PLATFORM_UI = platformUiFor(CURRENT_PLATFORM);

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

/** This machine's primary modifier, bound to the platform the shell detected. */
export function primaryModifierPressed(
  event: Pick<KeyboardEvent, "metaKey" | "ctrlKey" | "altKey">,
): boolean {
  return primaryModifierPressedOn(event, CURRENT_PLATFORM);
}
