import { beforeEach, describe, expect, it } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

import AppearanceCard from "./AppearanceCard.vue";
import { THEME_FAMILY_META, themeId } from "../../lib/theme";
import { useAppPrefsStore } from "../../stores/appPrefs";

beforeEach(() => {
  setActivePinia(createPinia());
});

/*
 * The theme cards and the tone control are the shared kit's `ThemePicker` now,
 * mounted here. What this suite still owns is that Look MOUNTS it, hands it
 * this app's stored choice, and applies what comes back to appPrefs — the
 * band's nested-`data-theme` substitution rule moved with the markup and is
 * pinned in `studio/components/settings/ThemePicker.test.ts`.
 */

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
    const tone = wrapper.get("[data-test='theme-tone']");
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
    await flushPromises();
    expect(patches.at(-1)).toEqual({ theme: "graphite-light" });

    // And the tone control moves the tone, keeping Nebula.
    const dark = wrapper
      .get("[data-test='theme-tone']")
      .findAll("button")
      .find((b) => b.text() === "Dark");
    await dark?.trigger("click");
    await flushPromises();
    expect(patches.at(-1)).toEqual({ theme: "nebula-dark" });
  });

  /*
   * `prefs.update` re-reads settings.json before merging, so two overlapping
   * calls both read the pre-change file and the second erases the first
   * field. Picking System moves BOTH theme and matchSystem, and the picker
   * reports them as two emits — one click must still be one write.
   */
  it("writes one patch when a choice moves both the theme and match-system", async () => {
    const prefs = useAppPrefsStore();
    prefs.settings = { theme: "nebula-dark", matchSystem: false } as never;
    const patches: unknown[] = [];
    prefs.update = async (patch) => void patches.push(patch);

    const wrapper = mount(AppearanceCard);
    const system = wrapper
      .get("[data-test='theme-tone']")
      .findAll("button")
      .find((b) => b.text() === "System");
    await system?.trigger("click");
    await flushPromises();

    expect(patches).toHaveLength(1);
    expect(patches[0]).toMatchObject({ matchSystem: true });
  });
});
