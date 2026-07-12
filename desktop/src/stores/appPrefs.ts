import { defineStore } from "pinia";
import {
  ipc,
  type AppSettings,
  type Theme,
  type ThemeFamily,
  type UpdateChannel,
} from "../lib/ipc";
import { applyUiScale, nextUiScale, type UiScaleDirection } from "../lib/uiScale";

/**
 * Resolve what `data-theme` should be on the root element. Pure — exported
 * for tests. `null` means "remove the attribute and let the system media
 * query drive the palette".
 */
export function resolveThemeAttributes(
  theme: Theme,
  family: ThemeFamily,
): { appearance: "light" | "dark" | null; family: ThemeFamily } {
  return {
    appearance: theme === "system" ? null : theme,
    family,
  };
}

function applyTheme(theme: Theme, family: ThemeFamily) {
  const attrs = resolveThemeAttributes(theme, family);
  const root = document.documentElement;
  if (attrs.appearance === null) delete root.dataset.theme;
  else root.dataset.theme = attrs.appearance;
  root.dataset.themeFamily = attrs.family;
}

/**
 * App-side preferences (settings.json via IPC): theme, notifications, dock
 * badge, engine env knobs. Loaded once at boot; every update persists and
 * re-applies immediately.
 */
export const useAppPrefsStore = defineStore("appPrefs", {
  state: () => ({
    settings: null as AppSettings | null,
  }),
  getters: {
    theme: (s): Theme => s.settings?.theme ?? "system",
    themeFamily: (s): ThemeFamily => s.settings?.themeFamily ?? "mold",
    notifications: (s) => s.settings?.notifications ?? true,
    dockBadge: (s) => s.settings?.dockBadge ?? true,
    restoreLastRoute: (s) => s.settings?.restoreLastRoute ?? false,
    runpodIncludeHfToken: (s) => s.settings?.runpodIncludeHfToken ?? false,
    runpodNetworkVolumeId: (s) => s.settings?.runpodNetworkVolumeId ?? null,
    uiScalePercent: (s) => s.settings?.uiScalePercent ?? 100,
    updateChannel: (s): UpdateChannel => s.settings?.updateChannel ?? "stable",
    engineEnv: (s): Record<string, string> => s.settings?.engineEnv ?? {},
  },
  actions: {
    async init(): Promise<AppSettings> {
      this.settings = await ipc.appSettingsGet();
      applyTheme(this.settings.theme, this.settings.themeFamily);
      const normalizedScale = await applyUiScale(this.settings.uiScalePercent);
      if (normalizedScale !== this.settings.uiScalePercent) {
        this.settings = { ...this.settings, uiScalePercent: normalizedScale };
        await ipc.appSettingsSet(this.settings);
      }
      return this.settings;
    },
    async update(patch: Partial<AppSettings>): Promise<void> {
      // Always merge onto the FRESH on-disk settings, never the in-memory
      // snapshot: other writers (the hosts store's saved-host persistence,
      // Rust's set_remote_host) write directly to settings.json, and
      // spreading a stale snapshot here would silently erase their fields
      // (saved hosts, reconnect list, remote mode) on the next theme toggle
      // or route change.
      const current = await ipc.appSettingsGet();
      this.settings = { ...current, ...patch };
      applyTheme(this.settings.theme, this.settings.themeFamily);
      const normalizedScale = await applyUiScale(this.settings.uiScalePercent);
      this.settings = { ...this.settings, uiScalePercent: normalizedScale };
      await ipc.appSettingsSet(this.settings);
    },
    async scaleUi(direction: UiScaleDirection): Promise<void> {
      const current = this.settings ?? (await this.init());
      await this.update({ uiScalePercent: nextUiScale(current.uiScalePercent, direction) });
    },
    /** Remember the route for restore-on-launch without churning the theme. */
    async rememberRoute(route: string): Promise<void> {
      if (!this.settings || this.settings.lastRoute === route) return;
      // Same fresh-merge rule as update() — this runs on every navigation.
      const current = await ipc.appSettingsGet();
      this.settings = { ...current, lastRoute: route };
      await ipc.appSettingsSet(this.settings);
    },
  },
});
