import { defineStore } from "pinia";
import { ipc, type AppSettings, type UpdateChannel } from "../lib/ipc";
import { applyUiScale, nextUiScale, type UiScaleDirection } from "../lib/uiScale";
import { normalizePanelWidth } from "../lib/panelResize";
import { DEFAULT_THEME, applyTheme, installSystemThemeSync, type ThemeId } from "../lib/theme";

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
    theme: (s): ThemeId => s.settings?.theme ?? DEFAULT_THEME,
    matchSystem: (s) => s.settings?.matchSystem ?? false,
    notifications: (s) => s.settings?.notifications ?? true,
    dockBadge: (s) => s.settings?.dockBadge ?? true,
    restoreLastRoute: (s) => s.settings?.restoreLastRoute ?? false,
    runpodIncludeHfToken: (s) => s.settings?.runpodIncludeHfToken ?? false,
    runpodNetworkVolumeId: (s) => s.settings?.runpodNetworkVolumeId ?? null,
    uiScalePercent: (s) => s.settings?.uiScalePercent ?? 100,
    updateChannel: (s): UpdateChannel => s.settings?.updateChannel ?? "stable",
    engineEnv: (s): Record<string, string> => s.settings?.engineEnv ?? {},
    saveRemoteOutputs: (s) => s.settings?.saveRemoteOutputs ?? true,
    mediaSaveDir: (s) => s.settings?.mediaSaveDir ?? null,
    navRailWidth: (s) => normalizePanelWidth("navRail", s.settings?.navRailWidth),
    generateParamsWidth: (s) =>
      normalizePanelWidth("generateParams", s.settings?.generateParamsWidth),
    historyDrawerWidth: (s) => normalizePanelWidth("historyDrawer", s.settings?.historyDrawerWidth),
    sidebarCollapsed: (s) => s.settings?.sidebarCollapsed ?? false,
  },
  actions: {
    async init(): Promise<AppSettings> {
      this.settings = await ipc.appSettingsGet();
      applyTheme(this.settings.theme, this.settings.matchSystem);
      // A system appearance flip re-resolves the pick without a settings write.
      installSystemThemeSync(() => ({ theme: this.theme, matchSystem: this.matchSystem }));
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
      // the boot remote-primary migration) write directly to settings.json,
      // and spreading a stale snapshot here would silently erase their fields
      // (saved hosts, reconnect list) on the next theme toggle or route change.
      const current = await ipc.appSettingsGet();
      this.settings = { ...current, ...patch };
      applyTheme(this.settings.theme, this.settings.matchSystem);
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
