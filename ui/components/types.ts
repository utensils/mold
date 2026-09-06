/*
 * Shared prop types for the kit's presenter components.
 *
 * These live in a plain module rather than inside their `.vue` files on
 * purpose: an SFC's named type exports are only visible to a toolchain whose
 * Vue language plugin resolves the SFC, and the apps also compile against a
 * `declare module "*.vue"` shim that exposes the default export alone. A
 * consumer importing `{ Toast }` straight from the SFC therefore type-checks
 * locally and fails in a clean sandbox build. Importing from here works
 * everywhere; the components re-export these names for convenience.
 */

/** One entry on the toast shelf. The host owns the list and its timers. */
export interface Toast {
  id: string;
  /** Severity: green success, yellow warning, red error; info stays neutral. */
  kind: "info" | "success" | "warning" | "error";
  text: string;
  actionLabel?: string;
}

/** One row in the ⌘K command palette, grouped by `section`. */
export interface PaletteItem {
  id: string;
  section: string;
  label: string;
  /** Trailing meta — the machine a model lives on, its load state, its family. */
  hint?: string;
}
