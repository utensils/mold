/**
 * Global keyboard map. The shell installs a single keydown listener and
 * resolves it here so the map stays testable as data.
 */

import { CURRENT_PLATFORM, type DesktopPlatform } from "./platform";

// The sidebar's destinations in order — ⌘1–⌘5 — plus ⌘, for Settings.
export const NAV_ROUTES: Readonly<Record<string, string>> = {
  "1": "/create",
  "2": "/queue",
  "3": "/library",
  "4": "/models",
  "5": "/machines",
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
  | { kind: "toggle-queue-pause" }
  | { kind: "ui-scale"; direction: "reset" | "in" | "out" };

export interface KeyLike {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
  repeat?: boolean;
}

/** What the shell knows when a key arrives with no modifier on it. */
export interface BareKeyContext {
  /** The element with focus, which may be entitled to the key itself. */
  target: Element | null;
  /** Whether a modal overlay owns the keyboard. */
  overlayOpen: boolean;
  /** The current route path. */
  route: string;
}

/**
 * The platform's plain Select All chord, no other modifiers: ⌘A on macOS,
 * Ctrl+A everywhere else. It is read for two jobs at once — suppressing the
 * webview's own Select All over app chrome, and driving Library's real
 * Select All — so a meta-only test does not merely skip a macOS paint quirk on
 * Windows and Linux, it leaves Ctrl+A doing nothing in the Library.
 */
export function isSelectAllChord(
  e: KeyLike,
  platform: DesktopPlatform = CURRENT_PLATFORM,
): boolean {
  if (e.altKey || e.shiftKey) return false;
  if (!(e.key === "a" || e.key === "A")) return false;
  return platform === "macos" ? e.metaKey && !e.ctrlKey : e.ctrlKey && !e.metaKey;
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
 * canvas — either opens the app's own menu or nothing, so the webview's
 * default Back / Reload / Inspect Element menu never appears on app chrome.
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

/**
 * Whether a bare key belongs to the focused element rather than the shell: a
 * field takes the character, and a button takes Space as its own activation.
 */
export function ownsBareKey(el: Element | null): boolean {
  if (!el) return false;
  const tag = el.tagName?.toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return true;
  if (tag === "button" || tag === "a" || tag === "summary") return true;
  if ((el as HTMLElement).isContentEditable) return true;
  return el.getAttribute?.("role") === "button";
}

/**
 * Whether a modal overlay owns the keyboard. Every kit panel that traps focus
 * marks itself `aria-modal`, so one query answers for all of them.
 */
export function overlayOwnsKeyboard(root: ParentNode = document): boolean {
  return root.querySelector('[aria-modal="true"]') !== null;
}

/**
 * The shell's one un-modified chord: Space pauses and resumes the queue
 * (status bar hint, README §3). It resolves only where a bare key is a
 * command — never inside a field or on a focused control, never under an
 * overlay, and never in My images, which spends Space on Quick Look.
 */
export function resolveBareShellShortcut(e: KeyLike, ctx: BareKeyContext): ShellAction | null {
  if (e.key !== " " || e.repeat) return null;
  if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return null;
  if (ctx.overlayOpen || ctx.route.startsWith("/library")) return null;
  return ownsBareKey(ctx.target) ? null : { kind: "toggle-queue-pause" };
}
