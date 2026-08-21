import { beforeEach, describe, expect, it } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useLibraryPrefsStore } from "./libraryPrefs";
import { useGenerateFormStore } from "./generateForm";
import {
  AUTO_TAG_TITLE_STORAGE_KEY,
  loadAutoTagTitle,
  saveAutoTagTitle,
  type LibraryPrefsStorage,
} from "../lib/libraryPrefs";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";

installMemoryLocalStorage();

function memoryStorage(seed: Record<string, string> = {}): LibraryPrefsStorage {
  const map = new Map(Object.entries(seed));
  return {
    getItem: (key) => map.get(key) ?? null,
    setItem: (key, value) => void map.set(key, value),
  };
}

beforeEach(() => {
  setActivePinia(createPinia());
  localStorage.removeItem(AUTO_TAG_TITLE_STORAGE_KEY);
});

describe("auto-tag-title persistence", () => {
  it("defaults on when nothing is stored", () => {
    expect(loadAutoTagTitle(memoryStorage())).toBe(true);
  });

  it("round-trips both values through storage", () => {
    const storage = memoryStorage();
    saveAutoTagTitle(false, storage);
    expect(storage.getItem(AUTO_TAG_TITLE_STORAGE_KEY)).toBe("false");
    expect(loadAutoTagTitle(storage)).toBe(false);
    saveAutoTagTitle(true, storage);
    expect(loadAutoTagTitle(storage)).toBe(true);
  });

  it("falls back to the default for an unrecognized value", () => {
    expect(loadAutoTagTitle(memoryStorage({ [AUTO_TAG_TITLE_STORAGE_KEY]: "yes" }))).toBe(true);
  });

  it("survives storage being unavailable", () => {
    expect(loadAutoTagTitle(null)).toBe(true);
    expect(() => saveAutoTagTitle(false, null)).not.toThrow();
  });
});

describe("libraryPrefs store", () => {
  it("mirrors the preference onto the Create form at boot", () => {
    localStorage.setItem(AUTO_TAG_TITLE_STORAGE_KEY, "false");
    const prefs = useLibraryPrefsStore();
    const form = useGenerateFormStore();
    prefs.init();
    expect(prefs.autoTagTitle).toBe(false);
    expect(form.form.fileUnderAutoTag).toBe(false);
  });

  it("persists and re-mirrors on every change", () => {
    const prefs = useLibraryPrefsStore();
    const form = useGenerateFormStore();
    prefs.init();
    expect(form.form.fileUnderAutoTag).toBe(true);
    prefs.setAutoTagTitle(false);
    expect(localStorage.getItem(AUTO_TAG_TITLE_STORAGE_KEY)).toBe("false");
    expect(form.form.fileUnderAutoTag).toBe(false);
    prefs.setAutoTagTitle(true);
    expect(form.form.fileUnderAutoTag).toBe(true);
  });
});
