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
      notifications: false,
      dockBadge: true,
      restoreLastRoute: true,
    }),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
  },
}));

import { ipc } from "../lib/ipc";
import { resolveThemeAttribute, useAppPrefsStore } from "./appPrefs";

describe("resolveThemeAttribute", () => {
  it("maps explicit themes and lets system remove the attribute", () => {
    expect(resolveThemeAttribute("dark")).toBe("dark");
    expect(resolveThemeAttribute("light")).toBe("light");
    expect(resolveThemeAttribute("system")).toBeNull();
  });
});

describe("appPrefs store", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    delete document.documentElement.dataset.theme;
  });

  it("init loads settings and stamps the theme on the root element", async () => {
    const prefs = useAppPrefsStore();
    await prefs.init();
    expect(prefs.theme).toBe("dark");
    expect(prefs.notifications).toBe(false);
    expect(prefs.engineEnv).toEqual({ MOLD_VAE_TILED: "force" });
    expect(document.documentElement.dataset.theme).toBe("dark");
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
