import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { THEME_FAMILY_META, themeId, type ThemeId } from "@ui/theme";
import ThemePicker from "./ThemePicker.vue";
import themePickerSource from "./ThemePicker.vue?raw";

function mountPicker(theme: ThemeId = "mocha-dark", matchSystem = false) {
  return mount(ThemePicker, { props: { theme, matchSystem } });
}

describe("ThemePicker cards", () => {
  it("paints each card's swatch band from that theme's own map", () => {
    const wrapper = mountPicker("safelight-light");
    for (const meta of THEME_FAMILY_META) {
      const band = wrapper.get(`[data-test='theme-${meta.id}'] [data-theme]`);
      // The band wears the theme it advertises, so ui/tokens.css stays the one
      // place a hex lives — the card never carries a colour of its own. It
      // paints in the tone currently in force, so the grid previews the choice
      // the person is actually about to make.
      expect(band.attributes("data-theme"), meta.id).toBe(
        themeId(meta.id, "light"),
      );
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
   * The band's cells must therefore read `var(--mold-*)` DIRECTLY, from scoped
   * rules on this component. Do not "clean these up" back into `bg-*`
   * utilities — that reintroduces the bug with no visible error. (Ported here
   * from AppearanceCard.test.ts with the band; it is the one themed island.)
   */
  it("reads --mold-* directly in the band, never a Tailwind colour alias", () => {
    const band = themePickerSource.match(
      /<span\s+:data-theme="themeId\([\s\S]*?<\/span>\s*<span class="ms-theme-card__label"/,
    )?.[0];
    expect(band, "the swatch band markup").toBeTruthy();
    expect(band).not.toMatch(/\bbg-(bg|bg-deep|surface|accent)\b/);

    const cellClasses = [
      ...(band ?? "").matchAll(/class="([^"]*ms-band__[^"]*)"/g),
    ].map((match) => match[1] ?? "");
    expect(cellClasses.length, "band cells").toBeGreaterThanOrEqual(4);

    for (const classes of cellClasses) {
      for (const name of classes
        .split(/\s+/)
        .filter((c) => c.startsWith("ms-band__"))) {
        const rule = themePickerSource.match(
          new RegExp(`\\.${name}\\s*\\{[^}]*\\}`, "s"),
        )?.[0];
        expect(rule, `scoped rule for .${name}`).toBeTruthy();
        expect(rule, `.${name} must paint from --mold-*`).toMatch(
          /var\(--mold-/,
        );
      }
    }
  });

  it("names the theme and never its tone", () => {
    // Tone lives on ONE control. A card that also said "dark" is what made
    // Match-system read as a second, contradicting theme picker.
    const wrapper = mountPicker();
    for (const meta of THEME_FAMILY_META) {
      const card = wrapper.get(`[data-test='theme-${meta.id}']`);
      expect(card.text(), meta.id).toContain(meta.label);
      expect(card.text(), meta.id).not.toMatch(/\b(dark|light)\b/i);
    }
  });

  it("marks the chosen family and no other", () => {
    const wrapper = mountPicker("blueprint-dark");
    const group = wrapper.get("[data-test='theme-select']");
    expect(group.attributes("role")).toBe("radiogroup");
    expect(
      wrapper.get("[data-test='theme-blueprint']").attributes("aria-checked"),
    ).toBe("true");
    expect(
      wrapper.get("[data-test='theme-mocha']").attributes("aria-checked"),
    ).toBe("false");
  });
});

describe("ThemePicker choices", () => {
  it("keeps the tone in force when a family is chosen", async () => {
    const wrapper = mountPicker("mocha-light");
    await wrapper.get("[data-test='theme-nebula']").trigger("click");
    expect(wrapper.emitted("update:theme")).toEqual([["nebula-light"]]);
    expect(wrapper.emitted("update:matchSystem")).toBeUndefined();
  });

  it("offers System, Light and Dark on one control and no Match-system switch", () => {
    const wrapper = mountPicker();
    const tone = wrapper.get("[data-test='theme-tone']");
    expect(tone.text()).toContain("System");
    expect(tone.text()).toContain("Light");
    expect(tone.text()).toContain("Dark");
    expect(wrapper.text()).not.toContain("Match system");
  });

  it("follows the machine when System is chosen, and stops when a tone is", async () => {
    const following = mountPicker("graphite-dark", false);
    const tone = following.findAllComponents({ name: "SegmentedControl" })[0]!;
    tone.vm.$emit("update:modelValue", "system");
    await following.vm.$nextTick();
    expect(following.emitted("update:matchSystem")).toEqual([[true]]);
    expect(following.emitted("update:theme")).toBeUndefined();

    const pinned = mountPicker("graphite-dark", true);
    pinned
      .findAllComponents({ name: "SegmentedControl" })[0]!
      .vm.$emit("update:modelValue", "light");
    await pinned.vm.$nextTick();
    expect(pinned.emitted("update:theme")).toEqual([["graphite-light"]]);
    expect(pinned.emitted("update:matchSystem")).toEqual([[false]]);
  });

  it("says what the tone in force means, in machine words", () => {
    expect(mountPicker("mocha-dark", true).text()).toContain(
      "Follows this machine",
    );
    expect(mountPicker("mocha-dark", false).text()).toContain(
      "whatever this machine does",
    );
  });
});
