import { invoke } from "@tauri-apps/api/core";
import { applyTheme, isTheme, isThemeFamily, type Theme, type ThemeFamily } from "../lib/theme";

export const MOBILE_SETTINGS_STORAGE_KEY = "mold.mobile.settings.v1";

export interface MobileSettings {
  theme: Theme;
  themeFamily: ThemeFamily;
}

export const DEFAULT_MOBILE_SETTINGS: Readonly<MobileSettings> = {
  theme: "system",
  themeFamily: "mold",
};

type SettingsStorage = Pick<Storage, "getItem" | "setItem">;
type MobileAppearanceInvoker = (command: string, args: { appearance: Theme }) => Promise<unknown>;

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

  try {
    await bridge("set_mobile_appearance", { appearance });
  } catch (error) {
    console.warn("Unable to synchronize the native mobile appearance", error);
  }
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
  };
  saveMobileSettings(next, storage);
  applyMobileSettings(next, nativeInvoke);
  return next;
}
