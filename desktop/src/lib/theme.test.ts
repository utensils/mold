import { afterEach, describe, expect, it } from "vitest";
import { applyTheme, isTheme, isThemeFamily, resolveThemeAttributes } from "./theme";

afterEach(() => {
  delete document.documentElement.dataset.theme;
  delete document.documentElement.dataset.themeFamily;
  document.documentElement.style.removeProperty("--bath");
  document.head.innerHTML = "";
});

describe("shared theme contract", () => {
  it("keeps appearance independent from the palette family", () => {
    expect(resolveThemeAttributes("dark", "mold")).toEqual({
      appearance: "dark",
      family: "mold",
    });
    expect(resolveThemeAttributes("system", "safelight")).toEqual({
      appearance: null,
      family: "safelight",
    });
  });

  it("validates only supported persisted values", () => {
    expect(isTheme("light")).toBe(true);
    expect(isTheme("sepia")).toBe(false);
    expect(isThemeFamily("mold")).toBe(true);
    expect(isThemeFamily("neon")).toBe(false);
  });

  it("applies root attributes and clears explicit appearance for System", () => {
    document.head.innerHTML = '<meta name="theme-color" content="#000000">';
    document.documentElement.style.setProperty("--bath", "#f7f5fa");

    applyTheme("light", "mold");
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(document.documentElement.dataset.themeFamily).toBe("mold");
    expect(document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')?.content).toBe(
      "#f7f5fa",
    );

    applyTheme("system", "safelight");
    expect(document.documentElement.dataset.theme).toBeUndefined();
    expect(document.documentElement.dataset.themeFamily).toBe("safelight");
  });
});
