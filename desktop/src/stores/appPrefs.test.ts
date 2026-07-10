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
      theme: "dark",
      themeFamily: "mold",
      notifications: false,
      dockBadge: true,
      restoreLastRoute: true,
    }),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
  },
}));

import { ipc } from "../lib/ipc";
import { resolveThemeAttributes, useAppPrefsStore } from "./appPrefs";

describe("resolveThemeAttributes", () => {
  it("maps appearance and family onto independent root attributes", () => {
    expect(resolveThemeAttributes("dark", "mold")).toEqual({
      appearance: "dark",
      family: "mold",
    });
    expect(resolveThemeAttributes("light", "safelight")).toEqual({
      appearance: "light",
      family: "safelight",
    });
    expect(resolveThemeAttributes("system", "mold")).toEqual({
      appearance: null,
      family: "mold",
    });
  });
});

describe("appPrefs store", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    delete document.documentElement.dataset.theme;
    delete document.documentElement.dataset.themeFamily;
  });

  it("defaults a fresh store to Mold with system appearance", () => {
    const prefs = useAppPrefsStore();
    expect(prefs.themeFamily).toBe("mold");
    expect(prefs.theme).toBe("system");
  });

  it("init loads settings and stamps the theme on the root element", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    expect(prefs.theme).toBe("dark");
    expect(prefs.themeFamily).toBe("mold");
    expect(prefs.notifications).toBe(false);
    expect(prefs.engineEnv).toEqual({ MOLD_VAE_TILED: "force" });
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(document.documentElement.dataset.themeFamily).toBe("mold");
  });

  it("switches theme family without changing appearance", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    await prefs.update({ themeFamily: "safelight" });
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(document.documentElement.dataset.themeFamily).toBe("safelight");
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenLastCalledWith(
      expect.objectContaining({ theme: "dark", themeFamily: "safelight" }),
    );
  });

  it("update persists and re-applies the theme; system clears the attribute", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    await prefs.update({ theme: "system" });
    expect(document.documentElement.dataset.theme).toBeUndefined();
    expect(vi.mocked(ipc.appSettingsSet)).toHaveBeenCalledWith(
      expect.objectContaining({ theme: "system" }),
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
