import { beforeEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

import AppearanceCard from "./AppearanceCard.vue";
import appearanceCardSource from "./AppearanceCard.vue?raw";
import { THEME_FAMILY_META, themeId } from "../../lib/theme";
import { useAppPrefsStore } from "../../stores/appPrefs";

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("Settings ▸ Look theme cards", () => {
  it("paints each card's swatch band from that theme's own map", () => {
    const wrapper = mount(AppearanceCard);
    const tone = useAppPrefsStore().theme.endsWith("-light") ? "light" : "dark";
    for (const meta of THEME_FAMILY_META) {
      const band = wrapper.get(`[data-test='theme-${meta.id}'] [data-theme]`);
      // The band wears the theme it advertises, so ui/tokens.css stays the one
      // place a hex lives — the card never carries a colour of its own. It
      // paints in the tone currently in force, so the grid previews the choice
      // the person is actually about to make.
      expect(band.attributes("data-theme"), meta.id).toBe(themeId(meta.id, tone));
      expect(band.attributes("style"), meta.id).toBeUndefined();
    }
  });

  /*
   * A Tailwind colour utility CANNOT paint a nested theme map.
   *
   * `bg-bg` resolves to `var(--color-bg)`, and `desktop/src/styles/tokens.css`
   * defines `--color-bg: var(--mold-bg)` inside `@theme`, i.e. on the ROOT.
   * A custom property's `var()` is substituted where the property is DEFINED,
   * not where it is used, and the SUBSTITUTED VALUE is what inherits down the
   * tree. So `--color-bg` inherits into the band already frozen to the root
   * theme's `--mold-bg`; the band's own `[data-theme="…"]` block redefines
   * `--mold-bg` far too late for anyone to read it. Every card painted the
   * theme the app was already wearing.
   *
   * The band's cells must therefore read `var(--mold-*)` DIRECTLY, from
   * scoped rules on this component. Do not "clean these up" back into `bg-*`
   * utilities — that reintroduces the bug with no visible error.
   */
  it("reads --mold-* directly in the band, never a Tailwind colour alias", () => {
    const band = appearanceCardSource.match(
      /<span\s+:data-theme="themeId\([\s\S]*?<\/span>\s*<span class="text-sm/,
    )?.[0];
    expect(band, "the swatch band markup").toBeTruthy();
    // No colour utility anywhere inside the band.
    expect(band).not.toMatch(/\bbg-(bg|bg-deep|surface|accent)\b/);

    const cellClasses = [...(band ?? "").matchAll(/class="([^"]*ms-band__[^"]*)"/g)].map(
      (m) => m[1] ?? "",
    );
    expect(cellClasses.length, "band cells").toBeGreaterThanOrEqual(4);

    // Every cell class the band uses is defined here and paints from --mold-*.
    for (const classes of cellClasses) {
      for (const name of classes.split(/\s+/).filter((c) => c.startsWith("ms-band__"))) {
        const rule = appearanceCardSource.match(new RegExp(`\\.${name}\\s*\\{[^}]*\\}`, "s"))?.[0];
        expect(rule, `scoped rule for .${name}`).toBeTruthy();
        expect(rule, `.${name} must paint from --mold-*`).toMatch(/var\(--mold-/);
      }
    }
  });

  it("names the theme and never its tone", () => {
    // Tone lives on ONE control. A card that also said "dark" is what made
    // Match-system read as a second, contradicting theme picker.
    const wrapper = mount(AppearanceCard);
    for (const meta of THEME_FAMILY_META) {
      const card = wrapper.get(`[data-test='theme-${meta.id}']`);
      expect(card.text(), meta.id).toContain(meta.label);
      expect(card.text(), meta.id).not.toMatch(/\b(dark|light)\b/i);
    }
  });

  it("offers System, Light and Dark on one control and no Match-system switch", () => {
    const wrapper = mount(AppearanceCard);
    const tone = wrapper.get("[data-test='tone-control']");
    expect(tone.text()).toContain("System");
    expect(tone.text()).toContain("Light");
    expect(tone.text()).toContain("Dark");
    expect(wrapper.text()).not.toContain("Match system");
  });

  it("changes the theme without changing the tone, and the tone without the theme", async () => {
    // `theme` is a getter over `settings`, and `update()` re-reads the file, so
    // assert the PATCH the card decides on — that is the component's contract.
    const prefs = useAppPrefsStore();
    prefs.settings = { theme: "nebula-light", matchSystem: false } as never;
    const patches: unknown[] = [];
    prefs.update = async (patch) => void patches.push(patch);

    const wrapper = mount(AppearanceCard);
    await wrapper.get("[data-test='theme-graphite']").trigger("click");
    expect(patches.at(-1)).toEqual({ theme: "graphite-light", matchSystem: false });

    // And the tone control moves the tone, keeping Nebula.
    const dark = wrapper
      .get("[data-test='tone-control']")
      .findAll("button")
      .find((b) => b.text() === "Dark");
    await dark?.trigger("click");
    expect(patches.at(-1)).toEqual({ theme: "nebula-dark", matchSystem: false });
  });
});
