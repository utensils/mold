import { beforeEach, describe, expect, it, vi } from "vitest";

/** theme.ts is a module singleton that reads storage at import time, so each
 * case resets modules and imports fresh against a prepared localStorage. */
async function importTheme() {
  vi.resetModules();
  return import("./theme");
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
});

describe("web theme defaults", () => {
  it("defaults fresh visitors to Safelight until the web redesign lands", async () => {
    const { theme, matchSystem } = await importTheme();
    expect(theme.value).toBe("safelight-dark");
    expect(matchSystem.value).toBe(false);
  });

  it("preserves a valid saved theme and its match-system flag", async () => {
    localStorage.setItem(
      "mold.web.theme.v1",
      JSON.stringify({ theme: "nebula-dark", matchSystem: true }),
    );
    const { theme, matchSystem } = await importTheme();
    expect(theme.value).toBe("nebula-dark");
    expect(matchSystem.value).toBe(true);
  });

  it("migrates the pre-tone ids, including the Porcelain merge", async () => {
    // Porcelain and Graphite were one theme under two names; the palette
    // survives as Graphite's light tone.
    for (const [saved, expected] of Object.entries({
      mocha: "mocha-dark",
      safelight: "safelight-dark",
      blueprint: "blueprint-light",
      graphite: "graphite-dark",
      porcelain: "graphite-light",
      nebula: "nebula-dark",
    })) {
      localStorage.setItem(
        "mold.web.theme.v1",
        JSON.stringify({ theme: saved }),
      );
      const { theme } = await importTheme();
      expect(theme.value, saved).toBe(expected);
    }
  });

  it("keeps Match system on while renaming a pre-tone theme", async () => {
    localStorage.setItem(
      "mold.web.theme.v1",
      JSON.stringify({ theme: "nebula", matchSystem: true }),
    );
    const { theme, matchSystem } = await importTheme();
    expect(theme.value).toBe("nebula-dark");
    expect(matchSystem.value).toBe(true);
  });

  it("migrates the pre-redesign family + appearance pair", async () => {
    localStorage.setItem(
      "mold.web.theme.v1",
      JSON.stringify({ family: "mold", theme: "light" }),
    );
    const { theme, matchSystem } = await importTheme();
    expect(theme.value).toBe("mocha-light");
    expect(matchSystem.value).toBe(false);
  });

  it("migrates a saved System appearance into match-system", async () => {
    localStorage.setItem(
      "mold.web.theme.v1",
      JSON.stringify({ family: "safelight", theme: "system" }),
    );
    const { theme, matchSystem } = await importTheme();
    expect(theme.value).toBe("safelight-dark");
    expect(matchSystem.value).toBe(true);
  });

  it("falls back to the default when the stored value is unrecognized", async () => {
    localStorage.setItem(
      "mold.web.theme.v1",
      JSON.stringify({ family: "vaporwave", theme: "sepia" }),
    );
    const { theme } = await importTheme();
    expect(theme.value).toBe("safelight-dark");
  });

  it("persists changes in the new shape and stamps one data-theme", async () => {
    const { theme, installTheme } = await importTheme();
    installTheme();
    theme.value = "graphite-dark";
    await Promise.resolve();
    expect(document.documentElement.dataset.theme).toBe("graphite-dark");
    expect(
      JSON.parse(localStorage.getItem("mold.web.theme.v1") ?? "{}"),
    ).toEqual({
      theme: "graphite-dark",
      matchSystem: false,
    });
  });
});
