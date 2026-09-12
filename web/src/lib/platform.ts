/*
 * The web SPA's platform binding. The conventions are shared with the desktop
 * shell in `@studio/lib/platform`; what lives here is how a BROWSER learns
 * which platform it is on, and the chord helpers bound to that answer.
 */
import {
  normalizePlatform,
  platformUi,
  primaryModifierPressed as primaryModifierPressedOn,
  type Platform,
} from "@studio/lib/platform";

/**
 * The platform behind a browser's `navigator.platform` (or the UA-Client-Hints
 * `userAgentData.platform`). iPhone, iPad and iPod are the Apple platform: an
 * iPad with a keyboard types ⌘ like every Mac, and none of those strings
 * contains "mac".
 */
export function detectWebPlatform(raw: string | undefined): Platform {
  const value = raw?.trim().toLowerCase();
  if (!value) return "unknown";
  if (
    value.includes("iphone") ||
    value.includes("ipad") ||
    value.includes("ipod") ||
    value.includes("ios")
  ) {
    return "macos";
  }
  return normalizePlatform(value);
}

function browserPlatform(): string | undefined {
  const nav = globalThis.navigator as
    (Navigator & { userAgentData?: { platform?: string } }) | undefined;
  return nav?.userAgentData?.platform ?? nav?.platform;
}

export const CURRENT_PLATFORM = detectWebPlatform(browserPlatform());

export const PLATFORM_UI = platformUi(CURRENT_PLATFORM);

/** This platform's spelling of a primary-modifier chord, e.g. `⌘E` / `Ctrl+E`. */
export function shortcutLabel(key: string): string {
  return `${PLATFORM_UI.modifierLabel}${key}`;
}

/** This platform's spelling of a primary-modifier + Shift chord, e.g. `⌘⇧N` / `Ctrl+Shift+N`. */
export function shiftShortcutLabel(key: string): string {
  return `${PLATFORM_UI.modifierLabel}${PLATFORM_UI.shiftLabel}${key}`;
}

/** This browser's primary modifier: Command on Apple platforms, Control elsewhere. */
export function primaryModifierPressed(
  event: Pick<KeyboardEvent, "metaKey" | "ctrlKey" | "altKey">,
): boolean {
  return primaryModifierPressedOn(event, CURRENT_PLATFORM);
}
