/**
 * Whether a mesh turntable exports over a transparent backdrop, remembered
 * between exports.
 *
 * A turntable of a 3-D object is the one export people drop onto a slide, a
 * README or a page that is not slate blue, so whether the backdrop comes with
 * it is a standing preference and not a per-export decision. It lives here
 * rather than in desktop's `AppSettings` for the reason `libraryPrefs.ts`
 * gives: `settings.json` is a Rust-owned struct, this choice never reaches the
 * engine, and the sheet it belongs to is shared by web, desktop and the phone.
 *
 * Not part of `VideoExportOptions`' defaults: an ordinary video re-encode has
 * no coverage to keep, and the server refuses `transparent` on one.
 */
export const TURNTABLE_TRANSPARENCY_STORAGE_KEY =
  "mold.export.turntableTransparent.v1";
export const TURNTABLE_TRANSPARENCY_DEFAULT = false;

export interface TurntableTransparencyStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function browserStorage(): TurntableTransparencyStorage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export function loadTurntableTransparency(
  storage: TurntableTransparencyStorage | null = browserStorage(),
): boolean {
  if (!storage) return TURNTABLE_TRANSPARENCY_DEFAULT;
  try {
    const saved = storage.getItem(TURNTABLE_TRANSPARENCY_STORAGE_KEY);
    if (saved === "true") return true;
    if (saved === "false") return false;
    return TURNTABLE_TRANSPARENCY_DEFAULT;
  } catch {
    return TURNTABLE_TRANSPARENCY_DEFAULT;
  }
}

export function saveTurntableTransparency(
  value: boolean,
  storage: TurntableTransparencyStorage | null = browserStorage(),
): void {
  if (!storage) return;
  try {
    storage.setItem(
      TURNTABLE_TRANSPARENCY_STORAGE_KEY,
      value ? "true" : "false",
    );
  } catch {
    // A browser that refuses to store the preference still exports.
  }
}
