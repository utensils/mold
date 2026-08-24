import { invoke } from "@tauri-apps/api/core";
import { applyTheme, isTheme, isThemeFamily, type Theme, type ThemeFamily } from "../lib/theme";

export const MOBILE_SETTINGS_STORAGE_KEY = "mold.mobile.settings.v1";

export interface MobileSettings {
  theme: Theme;
  themeFamily: ThemeFamily;
  autoSavePhotos: boolean;
  /** Settings ▸ Library "Tag new prints with their title" — the mirror the
   * Create form reads into `GenerateForm.fileUnderAutoTag`. On by default,
   * because the tag it files is always visible as the removable ghost chip
   * before Generate; a saved opt-out is preserved verbatim. */
  autoTagTitle: boolean;
}

export const DEFAULT_MOBILE_SETTINGS: Readonly<MobileSettings> = {
  theme: "dark",
  themeFamily: "safelight",
  autoSavePhotos: true,
  autoTagTitle: true,
};

type SettingsStorage = Pick<Storage, "getItem" | "setItem">;
type MobileAppearanceInvoker = (command: string, args: { appearance: Theme }) => Promise<unknown>;
type NativeAppearanceRequest = {
  appearance: Theme;
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
    return {
      theme: isTheme(parsed.theme) ? parsed.theme : DEFAULT_MOBILE_SETTINGS.theme,
      themeFamily: isThemeFamily(parsed.themeFamily)
        ? parsed.themeFamily
        : DEFAULT_MOBILE_SETTINGS.themeFamily,
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
  appearance: Theme,
  nativeInvoke?: MobileAppearanceInvoker,
): Promise<void> {
  const bridge =
    nativeInvoke ??
    ("__TAURI_INTERNALS__" in globalThis
      ? (command: string, args: { appearance: Theme }) => invoke(command, args)
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
  applyTheme(settings.theme, settings.themeFamily);
  void syncMobileNativeAppearance(settings.theme, nativeInvoke);
}

export function updateMobileSettings(
  current: MobileSettings,
  patch: Partial<MobileSettings>,
  storage: SettingsStorage | null = defaultStorage(),
  nativeInvoke?: MobileAppearanceInvoker,
): MobileSettings {
  const next: MobileSettings = {
    theme: isTheme(patch.theme) ? patch.theme : current.theme,
    themeFamily: isThemeFamily(patch.themeFamily) ? patch.themeFamily : current.themeFamily,
    autoSavePhotos:
      typeof patch.autoSavePhotos === "boolean" ? patch.autoSavePhotos : current.autoSavePhotos,
    autoTagTitle:
      typeof patch.autoTagTitle === "boolean" ? patch.autoTagTitle : current.autoTagTitle,
  };
  saveMobileSettings(next, storage);
  applyMobileSettings(next, nativeInvoke);
  return next;
}
