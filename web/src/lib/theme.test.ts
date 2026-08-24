import { beforeEach, describe, expect, it, vi } from "vitest";

/** theme.ts is a module singleton that reads storage at import time, so each
 * case resets modules and imports fresh against a prepared localStorage. */
async function importTheme() {
  vi.resetModules();
  return import("./theme");
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme-family");
  document.documentElement.removeAttribute("data-theme");
});

describe("web theme defaults", () => {
  it("defaults fresh visitors to Safelight Dark, like every other surface", async () => {
    const { themeFamily, theme } = await importTheme();
    expect(themeFamily.value).toBe("safelight");
    expect(theme.value).toBe("dark");
  });

  it("preserves a valid saved Mold choice", async () => {
    localStorage.setItem(
      "mold.web.theme.v1",
      JSON.stringify({ family: "mold", theme: "dark" }),
    );
    const { themeFamily, theme } = await importTheme();
    expect(themeFamily.value).toBe("mold");
    expect(theme.value).toBe("dark");
  });

  it("falls back to Safelight when the stored family is unrecognized", async () => {
    localStorage.setItem(
      "mold.web.theme.v1",
      JSON.stringify({ family: "vaporwave", theme: "light" }),
    );
    const { themeFamily, theme } = await importTheme();
    expect(themeFamily.value).toBe("safelight");
    expect(theme.value).toBe("light");
  });

  it("falls back to Dark when the stored appearance is unrecognized", async () => {
    localStorage.setItem(
      "mold.web.theme.v1",
      JSON.stringify({ family: "mold", theme: "sepia" }),
    );
    const { themeFamily, theme } = await importTheme();
    expect(themeFamily.value).toBe("mold");
    expect(theme.value).toBe("dark");
  });
});
