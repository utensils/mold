/**
 * Global keyboard map (macOS). Navigation uses ⌘1–⌘5 and ⌘, per the design
 * spec; the shell installs a single keydown listener and resolves it here so
 * the map stays testable as data.
 */

export const NAV_ROUTES: Readonly<Record<string, string>> = {
  "1": "/generate",
  "2": "/gallery",
  "3": "/chains",
  "4": "/models",
  "5": "/history",
  ",": "/settings",
};

export type ShellAction =
  | { kind: "navigate"; route: string }
  | { kind: "toggle-sidebar" }
  | { kind: "command-palette" }
  | { kind: "cancel-job" };

export interface KeyLike {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
}

/** Resolve a keydown into a shell-level action, or null if unhandled. */
export function resolveShellShortcut(e: KeyLike): ShellAction | null {
  if (!e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return null;
  const route = NAV_ROUTES[e.key];
  if (route) return { kind: "navigate", route };
  if (e.key === "\\") return { kind: "toggle-sidebar" };
  if (e.key === "k") return { kind: "command-palette" };
  if (e.key === ".") return { kind: "cancel-job" };
  return null;
}
