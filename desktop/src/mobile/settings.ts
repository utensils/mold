import { invoke } from "@tauri-apps/api/core";
import { THEME_TONE, applyTheme, isThemeId, migrateLegacyTheme, type ThemeId } from "../lib/theme";

/** What UIKit is told: follow the phone, or pin one trait. */
export type NativeAppearance = "system" | "dark" | "light";

export const MOBILE_SETTINGS_STORAGE_KEY = "mold.mobile.settings.v1";

export interface MobileSettings {
  theme: ThemeId;
  /** Follow the phone's appearance: paint `theme` or its light/dark partner. */
  matchSystem: boolean;
  autoSavePhotos: boolean;
  /** Settings ▸ Library "Tag new prints with their title" — the mirror the
   * Create form reads into `GenerateForm.fileUnderAutoTag`. On by default,
   * because the tag it files is always visible as the removable ghost chip
   * before Generate; a saved opt-out is preserved verbatim. */
  autoTagTitle: boolean;
}

export const DEFAULT_MOBILE_SETTINGS: Readonly<MobileSettings> = {
  theme: "safelight",
  matchSystem: false,
  autoSavePhotos: true,
  autoTagTitle: true,
};

type SettingsStorage = Pick<Storage, "getItem" | "setItem">;
type MobileAppearanceInvoker = (
  command: string,
  args: { appearance: NativeAppearance },
) => Promise<unknown>;
type NativeAppearanceRequest = {
  appearance: NativeAppearance;
  bridge: MobileAppearanceInvoker;
};

let pendingNativeAppearance: NativeAppearanceRequest | null = null;
let nativeAppearanceFlush: Promise<void> | null = null;

async function flushNativeAppearanceQueue(): Promise<void> {
  try {
    while (pendingNativeAppearance) {
      const next = pendingNativeAppearance;
      pendingNativeAppearance = null;
      try {
        await next.bridge("set_mobile_appearance", { appearance: next.appearance });
      } catch (error) {
        console.warn("Unable to synchronize the native mobile appearance", error);
      }
    }
  } finally {
    // Clear ownership before the flush promise resolves. A new request made
    // from another promise reaction can then start its own drain instead of
    // attaching to an already-finished flush and becoming stranded.
    nativeAppearanceFlush = null;
  }
}

function defaultStorage(): SettingsStorage | null {
  return typeof localStorage === "undefined" ? null : localStorage;
}

/** Load only known values, preserving a valid preference when its sibling is corrupt. */
export function loadMobileSettings(
  storage: SettingsStorage | null = defaultStorage(),
): MobileSettings {
  if (!storage) return { ...DEFAULT_MOBILE_SETTINGS };
  try {
    const parsed = JSON.parse(storage.getItem(MOBILE_SETTINGS_STORAGE_KEY) ?? "{}") as Record<
      string,
      unknown
    >;
    // A pre-redesign file carries `theme: system|dark|light` + `themeFamily`;
    // the shared table maps that pair onto a named theme.
    const migrated =
      isThemeId(parsed.theme) || parsed.theme === undefined
        ? null
        : migrateLegacyTheme(parsed.theme, parsed.themeFamily);
    return {
      theme:
        migrated?.theme ?? (isThemeId(parsed.theme) ? parsed.theme : DEFAULT_MOBILE_SETTINGS.theme),
      matchSystem:
        migrated?.matchSystem ??
        (typeof parsed.matchSystem === "boolean"
          ? parsed.matchSystem
          : DEFAULT_MOBILE_SETTINGS.matchSystem),
      autoSavePhotos:
        typeof parsed.autoSavePhotos === "boolean"
          ? parsed.autoSavePhotos
          : DEFAULT_MOBILE_SETTINGS.autoSavePhotos,
      // Absent on every install saved before File under shipped: that is the
      // migration, and it lands on the product default rather than off.
      autoTagTitle:
        typeof parsed.autoTagTitle === "boolean"
          ? parsed.autoTagTitle
          : DEFAULT_MOBILE_SETTINGS.autoTagTitle,
    };
  } catch {
    return { ...DEFAULT_MOBILE_SETTINGS };
  }
}

export function saveMobileSettings(
  settings: MobileSettings,
  storage: SettingsStorage | null = defaultStorage(),
): void {
  if (!storage) return;
  try {
    storage.setItem(MOBILE_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Appearance should still apply when WebKit storage is unavailable.
  }
}

/** Synchronize UIKit's trait so iOS chooses readable status-bar glyphs. */
export async function syncMobileNativeAppearance(
  appearance: NativeAppearance,
  nativeInvoke?: MobileAppearanceInvoker,
): Promise<void> {
  const bridge =
    nativeInvoke ??
    ("__TAURI_INTERNALS__" in globalThis
      ? (command: string, args: { appearance: NativeAppearance }) => invoke(command, args)
      : null);
  if (!bridge) return;

  pendingNativeAppearance = { appearance, bridge };
  if (!nativeAppearanceFlush) {
    nativeAppearanceFlush = flushNativeAppearanceQueue();
  }

  await nativeAppearanceFlush;
}

export function applyMobileSettings(
  settings: MobileSettings,
  nativeInvoke?: MobileAppearanceInvoker,
): void {
  applyTheme(settings.theme, settings.matchSystem);
  void syncMobileNativeAppearance(
    settings.matchSystem ? "system" : THEME_TONE[settings.theme],
    nativeInvoke,
  );
}

export function updateMobileSettings(
  current: MobileSettings,
  patch: Partial<MobileSettings>,
  storage: SettingsStorage | null = defaultStorage(),
  nativeInvoke?: MobileAppearanceInvoker,
): MobileSettings {
  const next: MobileSettings = {
    theme: isThemeId(patch.theme) ? patch.theme : current.theme,
    matchSystem: typeof patch.matchSystem === "boolean" ? patch.matchSystem : current.matchSystem,
    autoSavePhotos:
      typeof patch.autoSavePhotos === "boolean" ? patch.autoSavePhotos : current.autoSavePhotos,
    autoTagTitle:
      typeof patch.autoTagTitle === "boolean" ? patch.autoTagTitle : current.autoTagTitle,
  };
  saveMobileSettings(next, storage);
  applyMobileSettings(next, nativeInvoke);
  return next;
}
