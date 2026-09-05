import { describe, expect, it } from "vitest";
import { ICONS, ICON_NAMES, type IconName } from "./icons";

describe("icon registry", () => {
  it("ships the Library organization glyphs", () => {
    const organization = [
      "tag",
      "collection",
      "heart",
      "pencil",
    ] as const satisfies readonly IconName[];
    for (const name of organization) {
      expect(ICON_NAMES).toContain(name);
      expect(ICONS[name].length).toBeGreaterThan(0);
    }
  });

  it("ships a RunPod mark of its own rather than a generic cloud", () => {
    expect(ICON_NAMES).toContain("runpod");
    expect(ICONS).not.toHaveProperty("cloud");
  });

  it("keeps every glyph as stroke-only inner markup on the 24-unit grid", () => {
    for (const name of ICON_NAMES) {
      const markup = ICONS[name];
      expect(markup, name).not.toMatch(/<svg|<\/svg>/);
      expect(markup, name).not.toMatch(/fill="(?!none)/);
      expect(markup, name).not.toMatch(/stroke-width/);
      expect(markup, name).not.toMatch(/currentColor|#[0-9a-f]{3,6}\b/i);
    }
  });

  it("exposes ICON_NAMES in registry order", () => {
    expect(ICON_NAMES).toEqual(Object.keys(ICONS));
  });
});
