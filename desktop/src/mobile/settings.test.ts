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
  delete document.documentElement.dataset.themeFamily;
});

describe("mobile settings persistence", () => {
  it("defaults new and corrupt installs to Safelight, system appearance, and Photos auto-save", () => {
    expect(DEFAULT_MOBILE_SETTINGS).toEqual({
      theme: "system",
      themeFamily: "safelight",
      autoSavePhotos: true,
      autoTagTitle: true,
    });
    expect(loadMobileSettings(memoryStorage())).toEqual({
      theme: "system",
      themeFamily: "safelight",
      autoSavePhotos: true,
      autoTagTitle: true,
    });
    expect(loadMobileSettings(memoryStorage("not json"))).toEqual({
      theme: "system",
      themeFamily: "safelight",
      autoSavePhotos: true,
      autoTagTitle: true,
    });
  });

  it("keeps each valid field when another stored value is unknown", () => {
    expect(
      loadMobileSettings(
        memoryStorage(JSON.stringify({ theme: "light", themeFamily: "unknown", future: true })),
      ),
    ).toEqual({
      theme: "light",
      themeFamily: "safelight",
      autoSavePhotos: true,
      autoTagTitle: true,
    });
    expect(
      loadMobileSettings(
        memoryStorage(
          JSON.stringify({
            theme: "sepia",
            themeFamily: "safelight",
            autoSavePhotos: false,
          }),
        ),
      ),
    ).toEqual({
      theme: "system",
      themeFamily: "safelight",
      autoSavePhotos: false,
      autoTagTitle: true,
    });
  });

  it("preserves an existing user's saved Mold preference", () => {
    expect(
      loadMobileSettings(memoryStorage(JSON.stringify({ theme: "dark", themeFamily: "mold" }))),
    ).toEqual({
      theme: "dark",
      themeFamily: "mold",
      autoSavePhotos: true,
      autoTagTitle: true,
    });
  });

  it("persists and applies a change immediately, including native iOS appearance", () => {
    const storage = memoryStorage();
    const nativeInvoke = vi.fn().mockResolvedValue(undefined);
    const next = updateMobileSettings(
      { theme: "system", themeFamily: "mold", autoSavePhotos: true, autoTagTitle: true },
      { theme: "dark", themeFamily: "safelight" },
      storage,
      nativeInvoke,
    );

    expect(next).toEqual({
      theme: "dark",
      themeFamily: "safelight",
      autoSavePhotos: true,
      autoTagTitle: true,
    });
    expect(JSON.parse(storage.value() ?? "{}")).toEqual(next);
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(document.documentElement.dataset.themeFamily).toBe("safelight");
    expect(nativeInvoke).toHaveBeenCalledWith("set_mobile_appearance", {
      appearance: "dark",
    });
  });

  it("returns native iOS appearance control to the system", () => {
    const nativeInvoke = vi.fn().mockResolvedValue(undefined);

    updateMobileSettings(
      { theme: "dark", themeFamily: "mold", autoSavePhotos: true, autoTagTitle: true },
      { theme: "system" },
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
      { theme: "system", themeFamily: "safelight", autoSavePhotos: true, autoTagTitle: true },
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
      theme: "light",
      themeFamily: "mold",
      autoSavePhotos: false,
      autoTagTitle: true,
    });
  });

  it("persists an explicit auto-tag opt-out", () => {
    const storage = memoryStorage();
    const next = updateMobileSettings(
      { theme: "system", themeFamily: "safelight", autoSavePhotos: true, autoTagTitle: true },
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
