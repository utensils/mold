import { afterEach, describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import {
  THEMES,
  THEME_FAMILIES,
  THEME_FAMILY_META,
  applyFamilyChoice,
  applyTheme,
  applyToneChoice,
  familyOf,
  isThemeId,
  migrateLegacyTheme,
  partnerTheme,
  resolveTheme,
  themeId,
  toneChoice,
  toneOf,
  type ThemeId,
} from "./theme";

afterEach(() => {
  delete document.documentElement.dataset.theme;
  document.documentElement.style.removeProperty("--mold-bg-deep");
  document.head.innerHTML = "";
});

describe("shared theme contract", () => {
  it("gives every family both tones and nothing else", () => {
    expect(THEME_FAMILY_META.map((meta) => meta.id)).toEqual([...THEME_FAMILIES]);
    expect(THEMES).toHaveLength(THEME_FAMILIES.length * 2);
    for (const family of THEME_FAMILIES) {
      for (const tone of ["dark", "light"] as const) {
        const id = themeId(family, tone);
        expect(THEMES, id).toContain(id);
        expect(familyOf(id)).toBe(family);
        expect(toneOf(id)).toBe(tone);
      }
    }
  });

  it("never puts a tone in a theme's own label", () => {
    // The picker names the theme and the tone control names the tone; a label
    // that says "Mocha · dark" makes the two controls disagree by construction.
    // Word boundaries matter: "Safelight" and "darkroom" are names, not tones.
    for (const meta of THEME_FAMILY_META) {
      expect(meta.label, meta.id).not.toMatch(/\b(dark|light)\b/i);
      expect(meta.blurb, meta.id).not.toMatch(/\b(dark|light)\b/i);
      expect(meta.blurb, meta.id).toBeTruthy();
      expect(meta.type, meta.id).toContain(" · ");
    }
    // And the shape that produced "Mocha · dark" cannot come back: no field on
    // the metadata may hold a bare tone word.
    expect(Object.keys(THEME_FAMILY_META[0]!).sort()).toEqual(["blurb", "id", "label", "type"]);
  });

  it("pairs a theme with ITSELF in the other tone, never another family", () => {
    // The regression this whole contract exists for: Nebula used to become
    // Porcelain in daylight, which is a different typeface, radius and accent.
    for (const id of THEMES) {
      for (const tone of ["dark", "light"] as const) {
        const partner = partnerTheme(id, tone);
        expect(familyOf(partner), `${id} → ${tone}`).toBe(familyOf(id));
        expect(toneOf(partner), `${id} → ${tone}`).toBe(tone);
      }
      // A pick is its own partner on its own side.
      expect(partnerTheme(id, toneOf(id))).toBe(id);
    }
  });

  it("keeps the theme and the tone controls independent", () => {
    const saved = { theme: "nebula-dark" as ThemeId, matchSystem: false };

    // Choosing a tone never changes the family.
    expect(applyToneChoice("light", saved.theme)).toEqual({
      theme: "nebula-light",
      matchSystem: false,
    });
    // "System" keeps the stored tone as the fallback for a host that cannot
    // read the OS appearance, which is why both halves are persisted.
    expect(applyToneChoice("system", saved.theme)).toEqual({
      theme: "nebula-dark",
      matchSystem: true,
    });

    // Choosing a theme never changes the tone — including under System.
    expect(applyFamilyChoice("blueprint", { theme: "nebula-light", matchSystem: false })).toEqual({
      theme: "blueprint-light",
      matchSystem: false,
    });
    expect(applyFamilyChoice("graphite", { theme: "nebula-dark", matchSystem: true })).toEqual({
      theme: "graphite-dark",
      matchSystem: true,
    });
  });

  it("projects the persisted pair onto the three-position tone control", () => {
    expect(toneChoice({ theme: "mocha-dark", matchSystem: true })).toBe("system");
    expect(toneChoice({ theme: "mocha-dark", matchSystem: false })).toBe("dark");
    expect(toneChoice({ theme: "mocha-light", matchSystem: false })).toBe("light");
    // Round trip: every choice survives being applied and read back.
    for (const id of THEMES) {
      for (const choice of ["system", "light", "dark"] as const) {
        expect(toneChoice(applyToneChoice(choice, id)), `${id} ${choice}`).toBe(choice);
      }
    }
  });

  it("scopes every theme map to any element carrying data-theme, not only the root", () => {
    // The Look picker paints each card's swatch band from the theme's own map
    // by stamping `data-theme` on the band; a :root-only selector would leave
    // every band painted in the ACTIVE theme.
    const css = readFileSync("../ui/tokens.css", "utf8");
    for (const id of THEMES) {
      expect(css, id).toContain(`:root[data-theme="${id}"],\n[data-theme="${id}"] {`);
    }
  });

  it("declares exactly the themes ui/tokens.css carries", () => {
    const css = readFileSync("../ui/tokens.css", "utf8");
    const declared = [...css.matchAll(/:root\[data-theme="([\w-]+)"\]/g)].map((m) => m[1]);
    expect([...declared].sort()).toEqual([...THEMES].sort());
  });

  it("validates only the ten persisted ids", () => {
    expect(isThemeId("nebula-dark")).toBe(true);
    expect(isThemeId("nebula-light")).toBe(true);
    // The pre-tone ids and the retired name are migrated, never accepted.
    expect(isThemeId("nebula")).toBe(false);
    expect(isThemeId("porcelain")).toBe(false);
    expect(isThemeId("porcelain-light")).toBe(false);
    expect(isThemeId("dark")).toBe(false);
  });

  it("resolves a pick against the system appearance only when asked to", () => {
    expect(resolveTheme("mocha-dark", false, true)).toBe("mocha-dark");
    expect(resolveTheme("mocha-dark", true, true)).toBe("mocha-light");
    expect(resolveTheme("mocha-dark", true, false)).toBe("mocha-dark");
    // The old table sent these two to Porcelain, losing the theme entirely.
    expect(resolveTheme("nebula-dark", true, true)).toBe("nebula-light");
    expect(resolveTheme("safelight-dark", true, true)).toBe("safelight-light");
    for (const id of THEMES) {
      for (const light of [true, false]) {
        const once = resolveTheme(id, true, light);
        expect(resolveTheme(once, true, light), `${id} idempotent`).toBe(once);
      }
    }
  });

  it("migrates the pre-tone ids, including the Porcelain merge", () => {
    const table: Record<string, ThemeId> = {
      mocha: "mocha-dark",
      safelight: "safelight-dark",
      blueprint: "blueprint-light",
      graphite: "graphite-dark",
      porcelain: "graphite-light",
      nebula: "nebula-dark",
    };
    for (const [saved, expected] of Object.entries(table)) {
      expect(migrateLegacyTheme(saved, undefined), saved).toEqual({
        theme: expected,
        matchSystem: false,
      });
    }
  });

  it("keeps an explicit match-system flag while renaming the theme", () => {
    // The pre-tone shape carried the flag BESIDE the id, so migrating the id
    // must not answer for the flag. Losing it here silently turned Match
    // system off for everyone on upgrade — and only after first paint, since
    // the inline pre-paint script reads the flag directly.
    expect(migrateLegacyTheme("nebula", undefined, true)).toEqual({
      theme: "nebula-dark",
      matchSystem: true,
    });
    expect(migrateLegacyTheme("porcelain", undefined, true)).toEqual({
      theme: "graphite-light",
      matchSystem: true,
    });
    // A current id passes through with its flag too.
    expect(migrateLegacyTheme("mocha-light", undefined, true)).toEqual({
      theme: "mocha-light",
      matchSystem: true,
    });
    // Absent or false means false; the flag is never invented.
    expect(migrateLegacyTheme("nebula", undefined).matchSystem).toBe(false);
    expect(migrateLegacyTheme("nebula", undefined, false).matchSystem).toBe(false);
    // The oldest shape has no sibling flag — its appearance word decides.
    expect(migrateLegacyTheme("system", "mold", false)).toEqual({
      theme: "mocha-dark",
      matchSystem: true,
    });
  });

  it("migrates the legacy appearance + family pair in one hop", () => {
    expect(migrateLegacyTheme("dark", "safelight")).toEqual({
      theme: "safelight-dark",
      matchSystem: false,
    });
    // Safelight has its own light tone now; this used to land on Porcelain.
    expect(migrateLegacyTheme("light", "safelight")).toEqual({
      theme: "safelight-light",
      matchSystem: false,
    });
    expect(migrateLegacyTheme("system", "safelight")).toEqual({
      theme: "safelight-dark",
      matchSystem: true,
    });
    expect(migrateLegacyTheme("dark", "mold")).toEqual({
      theme: "mocha-dark",
      matchSystem: false,
    });
    expect(migrateLegacyTheme("light", "mold")).toEqual({
      theme: "mocha-light",
      matchSystem: false,
    });
    expect(migrateLegacyTheme("system", "mold")).toEqual({
      theme: "mocha-dark",
      matchSystem: true,
    });
    // Current values pass straight through; garbage lands on the legacy
    // default family rather than throwing.
    expect(migrateLegacyTheme("graphite-light", undefined)).toEqual({
      theme: "graphite-light",
      matchSystem: false,
    });
    expect(migrateLegacyTheme("sepia", "vaporwave")).toEqual({
      theme: "safelight-dark",
      matchSystem: false,
    });
  });

  it("stamps one resolved id on the root and syncs the chrome colour", () => {
    document.head.innerHTML = '<meta name="theme-color" content="#000000">';
    document.documentElement.style.setProperty("--mold-bg-deep", "#e6e9ef");

    expect(applyTheme("mocha-dark", true, document.documentElement, true)).toBe("mocha-light");
    expect(document.documentElement.dataset.theme).toBe("mocha-light");
    expect(document.documentElement.dataset.themeFamily).toBeUndefined();
    expect(document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')?.content).toBe(
      "#e6e9ef",
    );

    applyTheme("nebula-dark", false, document.documentElement, true);
    expect(document.documentElement.dataset.theme).toBe("nebula-dark");
  });
});
