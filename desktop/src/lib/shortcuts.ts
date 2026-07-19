/**
 * Global keyboard map. The shell installs a single keydown listener and
 * resolves it here so the map stays testable as data.
 */

import { CURRENT_PLATFORM, type DesktopPlatform } from "./platform";

export const NAV_ROUTES: Readonly<Record<string, string>> = {
  "1": "/generate",
  "2": "/gallery",
  "3": "/chains",
  "4": "/models",
  "5": "/history",
  "6": "/jobs",
  ",": "/settings",
};

export type ShellAction =
  | { kind: "navigate"; route: string }
  | { kind: "toggle-sidebar" }
  | { kind: "command-palette" }
  | { kind: "cancel-job" }
  | { kind: "new-generation" }
  | { kind: "randomize-seed" }
  | { kind: "copy-seed" }
  | { kind: "ui-scale"; direction: "reset" | "in" | "out" };

export interface KeyLike {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
}

/** Plain ⌘A (no other modifiers) — WebKit's Select All command. */
export function isSelectAllChord(e: KeyLike): boolean {
  return e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey && (e.key === "a" || e.key === "A");
}

/** Input types that hold selectable text (an unset `type` defaults to text). */
const TEXT_INPUT_TYPES = new Set(["text", "search", "url", "tel", "email", "password", "number"]);

/**
 * Whether the focused element should keep WebKit's native Select All.
 * `user-select: none` on <body> stops drag selection, but macOS WebKit still
 * honors ⌘A everywhere — which paints the whole app chrome as selected. The
 * shell intercepts ⌘A unless focus is genuinely editable or sits inside an
 * opted-in [data-selectable] region. Non-text inputs (checkbox, range, …)
 * are chrome, not text: they must not re-enable the global Select All.
 */
export function allowsNativeSelectAll(el: Element | null): boolean {
  if (!el) return false;
  const tag = el.tagName?.toLowerCase();
  if (tag === "input") return TEXT_INPUT_TYPES.has((el as HTMLInputElement).type || "text");
  if (tag === "textarea") return true;
  if ((el as HTMLElement).isContentEditable) return true;
  // isContentEditable above already covers contenteditable subtrees.
  return el.closest?.("[data-selectable]") != null;
}

/**
 * Whether a right-click target keeps the NATIVE context menu. Only genuine
 * text-editing surfaces qualify (spellcheck / paste); everything else — from
 * range sliders (`<input type="range">` is an input, but chrome) to the
 * canvas — either opens the app's own menu or nothing. The webview's default
 * Back / Reload / Inspect Element menu must never appear.
 */
export function allowsNativeContextMenu(el: Element | null): boolean {
  if (!el) return false;
  const tag = el.tagName?.toLowerCase();
  if (tag === "input") return TEXT_INPUT_TYPES.has((el as HTMLInputElement).type || "text");
  if (tag === "textarea") return true;
  return (el as HTMLElement).isContentEditable === true;
}

/**
 * Resolve a keydown into a shell-level action, or null if unhandled. Requires
 * the platform primary modifier and no Alt. Route-scoped actions (such as
 * randomize seed) are resolved here but gated by the current route in the shell.
 */
export function resolveShellShortcut(
  e: KeyLike,
  platform: DesktopPlatform = CURRENT_PLATFORM,
): ShellAction | null {
  const primaryPressed = platform === "macos" ? e.metaKey && !e.ctrlKey : e.ctrlKey && !e.metaKey;
  if (!primaryPressed || e.altKey) return null;
  // `+` is Shift+= on standard keyboards, so recognize zoom before the
  // general shifted-shortcut gate below.
  if (e.key === "+") return { kind: "ui-scale", direction: "in" };
  if (e.shiftKey) {
    return e.key === "c" || e.key === "C" ? { kind: "copy-seed" } : null;
  }
  const route = NAV_ROUTES[e.key];
  if (route) return { kind: "navigate", route };
  if (e.key === "\\") return { kind: "toggle-sidebar" };
  if (e.key === "k") return { kind: "command-palette" };
  if (e.key === ".") return { kind: "cancel-job" };
  if (e.key === "n") return { kind: "new-generation" };
  if (e.key === "r") return { kind: "randomize-seed" };
  if (e.key === "0") return { kind: "ui-scale", direction: "reset" };
  if (e.key === "=") return { kind: "ui-scale", direction: "in" };
  if (e.key === "-" || e.key === "_") return { kind: "ui-scale", direction: "out" };
  return null;
}
