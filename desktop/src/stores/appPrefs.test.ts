import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn().mockResolvedValue({
      mode: "local",
      remoteUrl: null,
      remoteApiKey: null,
      lastRoute: "/gallery",
      engineEnv: { MOLD_VAE_TILED: "force" },
      theme: "mocha",
      matchSystem: false,
      notifications: false,
      dockBadge: true,
      restoreLastRoute: true,
      runpodIncludeHfToken: true,
      runpodNetworkVolumeId: "nv-models",
      uiScalePercent: 120,
      updateChannel: "nightly",
      savedHosts: [],
      connectedHostIds: [],
      generateTargetHost: null,
      saveRemoteOutputs: true,
      navRailWidth: null,
      generateParamsWidth: null,
      historyDrawerWidth: null,
      sidebarCollapsed: false,
    }),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
  },
}));

import { ipc } from "../lib/ipc";
import { useAppPrefsStore } from "./appPrefs";

describe("appPrefs store", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    delete document.documentElement.dataset.theme;
  });

  it("defaults a fresh store to Mocha without system matching", () => {
    const prefs = useAppPrefsStore();
    expect(prefs.theme).toBe("mocha");
    expect(prefs.matchSystem).toBe(false);
    expect(prefs.updateChannel).toBe("stable");
  });

  it("init loads settings and stamps the theme on the root element", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    expect(prefs.theme).toBe("mocha");
    expect(prefs.matchSystem).toBe(false);
    expect(prefs.notifications).toBe(false);
    expect(prefs.engineEnv).toEqual({ MOLD_VAE_TILED: "force" });
    expect(prefs.runpodIncludeHfToken).toBe(true);
    expect(prefs.runpodNetworkVolumeId).toBe("nv-models");
    expect(prefs.uiScalePercent).toBe(120);
    expect(prefs.updateChannel).toBe("nightly");
    expect(document.documentElement.dataset.theme).toBe("mocha");
  });

  it("persists whole-app scaling", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    await prefs.scaleUi("in");
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({ uiScalePercent: 130 }),
    );
  });

  it("persists the selected RunPod network volume", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    await prefs.update({ runpodNetworkVolumeId: "nv-renders" });
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({ runpodNetworkVolumeId: "nv-renders" }),
    );
  });

  it("persists the RunPod Hugging Face token preference", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    await prefs.update({ runpodIncludeHfToken: false });
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({ runpodIncludeHfToken: false }),
    );
  });

  it("persists the selected desktop update channel", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    await prefs.update({ updateChannel: "stable" });
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({ updateChannel: "stable" }),
    );
  });

  it("switches theme family without changing appearance", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    await prefs.update({ theme: "safelight" });
    expect(document.documentElement.dataset.theme).toBe("safelight");
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({ theme: "safelight", matchSystem: false }),
    );
  });

  it("update persists and re-applies the theme; match-system follows the OS", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: query.includes("light"),
      addEventListener: () => {},
      removeEventListener: () => {},
    }));
    await prefs.update({ matchSystem: true });
    // A light system appearance paints the pick's light partner.
    expect(document.documentElement.dataset.theme).toBe("blueprint");
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenCalledWith(
      expect.objectContaining({ matchSystem: true }),
    );
  });

  it("rememberRoute persists without churn when unchanged", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    const calls = vi.mocked(ipc.appSettingsSet).mock.calls.length;
    await prefs.rememberRoute("/gallery"); // unchanged
    expect(vi.mocked(ipc.appSettingsSet).mock.calls.length).toBe(calls);
    await prefs.rememberRoute("/models");
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({ lastRoute: "/models" }),
    );
  });
});

describe("appPrefs concurrent-writer safety", () => {
  it("update() merges onto fresh disk settings, not the boot snapshot", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init(); // snapshot has savedHosts: []
    // Another writer (the hosts store's saved-host persistence) persists a host…
    vi.mocked(ipc.appSettingsGet).mockResolvedValue({
      ...(prefs.settings as NonNullable<typeof prefs.settings>),
      savedHosts: [
        { id: "hal9000-7680", name: "hal9000", url: "http://hal9000:7680", lastUsedMs: 1 },
      ],
      connectedHostIds: ["hal9000-7680"],
    });
    // …then a routine pref write happens. It must NOT erase the host.
    await prefs.update({ theme: "porcelain" });
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({
        theme: "porcelain",
        connectedHostIds: ["hal9000-7680"],
        savedHosts: [expect.objectContaining({ id: "hal9000-7680" })],
      }),
    );
  });

  it("rememberRoute() also merges onto fresh disk settings", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    vi.mocked(ipc.appSettingsGet).mockResolvedValue({
      ...(prefs.settings as NonNullable<typeof prefs.settings>),
      connectedHostIds: ["hal9000-7680"],
    });
    await prefs.rememberRoute("/jobs");
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({ lastRoute: "/jobs", connectedHostIds: ["hal9000-7680"] }),
    );
  });
});

describe("appPrefs panel widths", () => {
  function panelSettings(overrides: Record<string, unknown> = {}) {
    return {
      mode: "local",
      remoteUrl: null,
      remoteApiKey: null,
      lastRoute: null,
      engineEnv: {},
      theme: "mocha",
      matchSystem: true,
      notifications: true,
      dockBadge: true,
      restoreLastRoute: false,
      runpodIncludeHfToken: false,
      runpodNetworkVolumeId: null,
      uiScalePercent: 100,
      updateChannel: "stable",
      savedHosts: [],
      connectedHostIds: [],
      generateTargetHost: null,
      saveRemoteOutputs: true,
      navRailWidth: null,
      generateParamsWidth: null,
      sidebarCollapsed: false,
      ...overrides,
    };
  }

  beforeEach(() => {
    setActivePinia(createPinia());
  });

  it("defaults to the PANEL_LIMITS defaults before settings load", () => {
    const prefs = useAppPrefsStore();
    expect(prefs.navRailWidth).toBe(270);
    expect(prefs.generateParamsWidth).toBe(340);
    expect(prefs.historyDrawerWidth).toBe(290);
  });

  it("defaults to the PANEL_LIMITS defaults when the persisted values are null", async () => {
    vi.mocked(ipc.appSettingsGet).mockResolvedValue(panelSettings() as never);
    const prefs = useAppPrefsStore();
    await prefs.init();
    expect(prefs.navRailWidth).toBe(270);
    expect(prefs.generateParamsWidth).toBe(340);
    expect(prefs.historyDrawerWidth).toBe(290);
  });

  it("reflects persisted widths", async () => {
    vi.mocked(ipc.appSettingsGet).mockResolvedValue(
      panelSettings({
        navRailWidth: 260,
        generateParamsWidth: 400,
        historyDrawerWidth: 700,
      }) as never,
    );
    const prefs = useAppPrefsStore();
    await prefs.init();
    expect(prefs.navRailWidth).toBe(260);
    expect(prefs.generateParamsWidth).toBe(400);
    expect(prefs.historyDrawerWidth).toBe(700);
  });

  it("clamps absurd persisted widths into the panel limits", async () => {
    vi.mocked(ipc.appSettingsGet).mockResolvedValue(
      panelSettings({
        navRailWidth: 9000,
        generateParamsWidth: 4,
        historyDrawerWidth: 9000,
      }) as never,
    );
    const prefs = useAppPrefsStore();
    await prefs.init();
    expect(prefs.navRailWidth).toBe(360);
    expect(prefs.generateParamsWidth).toBe(280);
    expect(prefs.historyDrawerWidth).toBe(960);
  });

  it("defaults the sidebar to expanded and persists a collapse toggle", async () => {
    vi.mocked(ipc.appSettingsGet).mockResolvedValue(panelSettings() as never);
    const prefs = useAppPrefsStore();
    expect(prefs.sidebarCollapsed).toBe(false);
    await prefs.init();
    expect(prefs.sidebarCollapsed).toBe(false);
    await prefs.update({ sidebarCollapsed: true });
    expect(prefs.sidebarCollapsed).toBe(true);
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({ sidebarCollapsed: true }),
    );
  });

  it("persists a committed width and a null reset through update()", async () => {
    vi.mocked(ipc.appSettingsGet).mockResolvedValue(panelSettings() as never);
    const prefs = useAppPrefsStore();
    await prefs.init();
    await prefs.update({ navRailWidth: 260 });
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({ navRailWidth: 260 }),
    );
    await prefs.update({ generateParamsWidth: null });
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({ generateParamsWidth: null }),
    );
  });
});
