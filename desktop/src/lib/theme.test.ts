import { afterEach, describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import {
  THEMES,
  THEME_META,
  THEME_PAIR,
  THEME_TONE,
  applyTheme,
  isThemeId,
  migrateLegacyTheme,
  resolveTheme,
} from "./theme";

afterEach(() => {
  delete document.documentElement.dataset.theme;
  document.documentElement.style.removeProperty("--mold-bg-deep");
  document.head.innerHTML = "";
});

describe("shared theme contract", () => {
  it("names every theme exactly once in the metadata, tone, and pairing tables", () => {
    expect(THEME_META.map((meta) => meta.id)).toEqual([...THEMES]);
    for (const id of THEMES) {
      expect(THEME_TONE[id]).toBe(THEME_META.find((meta) => meta.id === id)?.tone);
      expect(THEME_TONE[THEME_PAIR[id].dark], `${id} dark partner`).toBe("dark");
      expect(THEME_TONE[THEME_PAIR[id].light], `${id} light partner`).toBe("light");
      // A pick keeps itself on its own side of the pairing.
      expect(THEME_PAIR[id][THEME_TONE[id]]).toBe(id);
    }
  });

  it("declares exactly the themes ui/tokens.css carries", () => {
    const css = readFileSync("../ui/tokens.css", "utf8");
    const declared = [...css.matchAll(/:root\[data-theme="([\w-]+)"\]/g)].map((m) => m[1]);
    expect([...declared].sort()).toEqual([...THEMES].sort());
  });

  it("validates only the six persisted ids", () => {
    expect(isThemeId("nebula")).toBe(true);
    expect(isThemeId("dark")).toBe(false);
    expect(isThemeId("mold")).toBe(false);
  });

  it("resolves a pick against the system appearance only when asked to", () => {
    expect(resolveTheme("mocha", false, true)).toBe("mocha");
    expect(resolveTheme("mocha", true, true)).toBe("blueprint");
    expect(resolveTheme("mocha", true, false)).toBe("mocha");
    expect(resolveTheme("porcelain", true, false)).toBe("graphite");
    for (const id of THEMES) {
      for (const light of [true, false]) {
        const once = resolveTheme(id, true, light);
        expect(resolveTheme(once, true, light), `${id} idempotent`).toBe(once);
      }
    }
  });

  it("migrates the legacy appearance + family pair through one table", () => {
    expect(migrateLegacyTheme("dark", "safelight")).toEqual({
      theme: "safelight",
      matchSystem: false,
    });
    expect(migrateLegacyTheme("light", "safelight")).toEqual({
      theme: "porcelain",
      matchSystem: false,
    });
    expect(migrateLegacyTheme("system", "safelight")).toEqual({
      theme: "safelight",
      matchSystem: true,
    });
    expect(migrateLegacyTheme("dark", "mold")).toEqual({ theme: "mocha", matchSystem: false });
    expect(migrateLegacyTheme("light", "mold")).toEqual({
      theme: "blueprint",
      matchSystem: false,
    });
    expect(migrateLegacyTheme("system", "mold")).toEqual({ theme: "mocha", matchSystem: true });
    // Already migrated values pass straight through; garbage lands on the
    // legacy default family rather than throwing.
    expect(migrateLegacyTheme("graphite", undefined)).toEqual({
      theme: "graphite",
      matchSystem: false,
    });
    expect(migrateLegacyTheme("sepia", "vaporwave")).toEqual({
      theme: "safelight",
      matchSystem: false,
    });
  });

  it("stamps one resolved id on the root and syncs the chrome colour", () => {
    document.head.innerHTML = '<meta name="theme-color" content="#000000">';
    document.documentElement.style.setProperty("--mold-bg-deep", "#e9eff7");

    expect(applyTheme("mocha", true, document.documentElement, true)).toBe("blueprint");
    expect(document.documentElement.dataset.theme).toBe("blueprint");
    expect(document.documentElement.dataset.themeFamily).toBeUndefined();
    expect(document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')?.content).toBe(
      "#e9eff7",
    );

    applyTheme("nebula", false, document.documentElement, true);
    expect(document.documentElement.dataset.theme).toBe("nebula");
  });
});
