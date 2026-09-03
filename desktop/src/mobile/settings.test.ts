import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_MOBILE_SETTINGS,
  MOBILE_SETTINGS_STORAGE_KEY,
  loadMobileSettings,
  syncMobileNativeAppearance,
  updateMobileSettings,
} from "./settings";

function memoryStorage(initial?: string) {
  const values = new Map<string, string>();
  if (initial !== undefined) values.set(MOBILE_SETTINGS_STORAGE_KEY, initial);
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    value: () => values.get(MOBILE_SETTINGS_STORAGE_KEY) ?? null,
  };
}

afterEach(() => {
  delete document.documentElement.dataset.theme;
});

describe("mobile settings persistence", () => {
  it("defaults new and corrupt installs to Safelight and Photos auto-save", () => {
    expect(DEFAULT_MOBILE_SETTINGS).toEqual({
      theme: "safelight",
      matchSystem: false,
      autoSavePhotos: true,
      autoTagTitle: true,
    });
    expect(loadMobileSettings(memoryStorage())).toEqual({
      theme: "safelight",
      matchSystem: false,
      autoSavePhotos: true,
      autoTagTitle: true,
    });
    expect(loadMobileSettings(memoryStorage("not json"))).toEqual({
      theme: "safelight",
      matchSystem: false,
      autoSavePhotos: true,
      autoTagTitle: true,
    });
  });

  it("keeps each valid field when another stored value is unknown", () => {
    expect(
      loadMobileSettings(
        memoryStorage(JSON.stringify({ theme: "graphite", matchSystem: "yes", future: true })),
      ),
    ).toEqual({
      theme: "graphite",
      matchSystem: false,
      autoSavePhotos: true,
      autoTagTitle: true,
    });
    expect(
      loadMobileSettings(memoryStorage(JSON.stringify({ theme: "sepia", autoSavePhotos: false }))),
    ).toEqual({
      theme: "safelight",
      matchSystem: false,
      autoSavePhotos: false,
      autoTagTitle: true,
    });
  });

  it("migrates the pre-redesign family + appearance pair through the shared table", () => {
    expect(
      loadMobileSettings(memoryStorage(JSON.stringify({ theme: "dark", themeFamily: "mold" }))),
    ).toEqual({
      theme: "mocha",
      matchSystem: false,
      autoSavePhotos: true,
      autoTagTitle: true,
    });
    expect(
      loadMobileSettings(
        memoryStorage(JSON.stringify({ theme: "system", themeFamily: "safelight" })),
      ),
    ).toEqual({
      theme: "safelight",
      matchSystem: true,
      autoSavePhotos: true,
      autoTagTitle: true,
    });
  });

  it("persists and applies a change immediately, including native iOS appearance", () => {
    const storage = memoryStorage();
    const nativeInvoke = vi.fn().mockResolvedValue(undefined);
    const next = updateMobileSettings(
      { theme: "mocha", matchSystem: true, autoSavePhotos: true, autoTagTitle: true },
      { theme: "porcelain", matchSystem: false },
      storage,
      nativeInvoke,
    );

    expect(next).toEqual({
      theme: "porcelain",
      matchSystem: false,
      autoSavePhotos: true,
      autoTagTitle: true,
    });
    expect(JSON.parse(storage.value() ?? "{}")).toEqual(next);
    expect(document.documentElement.dataset.theme).toBe("porcelain");
    // UIKit is told the painted theme's tone so status-bar glyphs stay readable.
    expect(nativeInvoke).toHaveBeenCalledWith("set_mobile_appearance", {
      appearance: "light",
    });
  });

  it("returns native iOS appearance control to the system", () => {
    const nativeInvoke = vi.fn().mockResolvedValue(undefined);

    updateMobileSettings(
      { theme: "mocha", matchSystem: false, autoSavePhotos: true, autoTagTitle: true },
      { matchSystem: true },
      memoryStorage(),
      nativeInvoke,
    );

    expect(nativeInvoke).toHaveBeenCalledWith("set_mobile_appearance", {
      appearance: "system",
    });
  });

  it("persists an explicit Photos auto-save preference", () => {
    const storage = memoryStorage();
    const next = updateMobileSettings(
      { theme: "safelight", matchSystem: true, autoSavePhotos: true, autoTagTitle: true },
      { autoSavePhotos: false },
      storage,
    );

    expect(next.autoSavePhotos).toBe(false);
    expect(JSON.parse(storage.value() ?? "{}")).toEqual(next);
  });

  it("keeps a saved auto-tag opt-out across a reload", () => {
    // The preference is a mirror of the visible ghost chip: an existing user
    // who turned it off must not have it silently restored by an upgrade.
    expect(
      loadMobileSettings(memoryStorage(JSON.stringify({ autoTagTitle: false }))).autoTagTitle,
    ).toBe(false);
    expect(
      loadMobileSettings(memoryStorage(JSON.stringify({ autoTagTitle: "no" }))).autoTagTitle,
    ).toBe(true);
  });

  it("migrates an install saved before File under to tagging prints with their title", () => {
    // A pre-File-under blob has no `autoTagTitle` key at all; the product
    // default is on, and every other saved choice survives the migration.
    const saved = loadMobileSettings(
      memoryStorage(JSON.stringify({ theme: "light", themeFamily: "mold", autoSavePhotos: false })),
    );

    expect(saved).toEqual({
      theme: "blueprint",
      matchSystem: false,
      autoSavePhotos: false,
      autoTagTitle: true,
    });
  });

  it("persists an explicit auto-tag opt-out", () => {
    const storage = memoryStorage();
    const next = updateMobileSettings(
      { theme: "safelight", matchSystem: true, autoSavePhotos: true, autoTagTitle: true },
      { autoTagTitle: false },
      storage,
    );

    expect(next.autoTagTitle).toBe(false);
    expect(JSON.parse(storage.value() ?? "{}")).toEqual(next);
  });

  it("serializes native updates and applies only the latest pending appearance", async () => {
    let finishFirst: (() => void) | undefined;
    const firstUpdate = new Promise<void>((resolve) => {
      finishFirst = resolve;
    });
    const nativeInvoke = vi.fn().mockReturnValueOnce(firstUpdate).mockResolvedValue(undefined);

    const darkUpdate = syncMobileNativeAppearance("dark", nativeInvoke);
    const lightUpdate = syncMobileNativeAppearance("light", nativeInvoke);
    const systemUpdate = syncMobileNativeAppearance("system", nativeInvoke);

    expect(nativeInvoke).toHaveBeenCalledTimes(1);
    expect(nativeInvoke).toHaveBeenNthCalledWith(1, "set_mobile_appearance", {
      appearance: "dark",
    });

    finishFirst?.();
    await Promise.all([darkUpdate, lightUpdate, systemUpdate]);

    expect(nativeInvoke).toHaveBeenCalledTimes(2);
    expect(nativeInvoke).toHaveBeenNthCalledWith(2, "set_mobile_appearance", {
      appearance: "system",
    });
  });

  it("starts a new native flush for a request made as the prior bridge resolves", async () => {
    let finishFirst: (() => void) | undefined;
    const firstUpdate = new Promise<void>((resolve) => {
      finishFirst = resolve;
    });
    const nativeInvoke = vi.fn().mockReturnValueOnce(firstUpdate).mockResolvedValue(undefined);

    const darkUpdate = syncMobileNativeAppearance("dark", nativeInvoke);
    const lightUpdate = firstUpdate.then(() => syncMobileNativeAppearance("light", nativeInvoke));

    finishFirst?.();
    await Promise.all([darkUpdate, lightUpdate]);

    expect(nativeInvoke).toHaveBeenCalledTimes(2);
    expect(nativeInvoke).toHaveBeenLastCalledWith("set_mobile_appearance", {
      appearance: "light",
    });
  });
});
