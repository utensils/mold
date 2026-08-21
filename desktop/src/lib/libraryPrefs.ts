/**
 * Desktop-owned Library preferences that are NOT engine config.
 *
 * `gallery.trash_retention_days` belongs to the host and lives on
 * `/api/config`; "Tag new prints with their title" is a property of this
 * install's Create form, so it is stored in the webview beside the other
 * client-side view preferences (the same place the Library thumbnail-size
 * target lives). It is deliberately NOT an `AppSettings` field: settings.json
 * is a Rust-owned struct and this preference never reaches the engine.
 */

/** Storage key for the "Tag new prints with their title" toggle. */
export const AUTO_TAG_TITLE_STORAGE_KEY = "library.autoTagTitle";

/** The product default: a titled print picks up its own slug as a tag. */
export const AUTO_TAG_TITLE_DEFAULT = true;

export interface LibraryPrefsStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function browserStorage(): LibraryPrefsStorage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

/** Read the toggle. Anything unrecognized (absent, corrupt, another app's
 * value) falls back to the default rather than silently filing nothing. */
export function loadAutoTagTitle(storage: LibraryPrefsStorage | null = browserStorage()): boolean {
  if (!storage) return AUTO_TAG_TITLE_DEFAULT;
  try {
    const saved = storage.getItem(AUTO_TAG_TITLE_STORAGE_KEY);
    if (saved === "true") return true;
    if (saved === "false") return false;
    return AUTO_TAG_TITLE_DEFAULT;
  } catch {
    return AUTO_TAG_TITLE_DEFAULT;
  }
}

/** Persist the toggle; a storage failure is never fatal to the session. */
export function saveAutoTagTitle(
  value: boolean,
  storage: LibraryPrefsStorage | null = browserStorage(),
): void {
  if (!storage) return;
  try {
    storage.setItem(AUTO_TAG_TITLE_STORAGE_KEY, value ? "true" : "false");
  } catch {
    /* private mode / quota — the session keeps the value in memory */
  }
}
