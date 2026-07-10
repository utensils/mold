import { defineStore } from "pinia";
import { ipc, type AppSettings, type Theme } from "../lib/ipc";

/**
 * Resolve what `data-theme` should be on the root element. Pure — exported
 * for tests. `null` means "remove the attribute and let the system media
 * query drive the palette".
 */
export function resolveThemeAttribute(theme: Theme): "light" | "dark" | null {
  return theme === "system" ? null : theme;
}

function applyTheme(theme: Theme) {
  const attr = resolveThemeAttribute(theme);
  const root = document.documentElement;
  if (attr === null) delete root.dataset.theme;
  else root.dataset.theme = attr;
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
    notifications: (s) => s.settings?.notifications ?? true,
    dockBadge: (s) => s.settings?.dockBadge ?? true,
    restoreLastRoute: (s) => s.settings?.restoreLastRoute ?? false,
    engineEnv: (s): Record<string, string> => s.settings?.engineEnv ?? {},
  },
  actions: {
    async init(): Promise<AppSettings> {
      this.settings = await ipc.appSettingsGet();
      applyTheme(this.settings.theme);
      return this.settings;
    },
    async update(patch: Partial<AppSettings>): Promise<void> {
      const current = this.settings ?? (await ipc.appSettingsGet());
      this.settings = { ...current, ...patch };
      applyTheme(this.settings.theme);
      await ipc.appSettingsSet(this.settings);
    },
    /** Remember the route for restore-on-launch without churning the theme. */
    async rememberRoute(route: string): Promise<void> {
      if (!this.settings || this.settings.lastRoute === route) return;
      this.settings = { ...this.settings, lastRoute: route };
      await ipc.appSettingsSet(this.settings);
    },
  },
});
