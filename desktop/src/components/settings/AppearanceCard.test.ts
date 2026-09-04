import { beforeEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

import AppearanceCard from "./AppearanceCard.vue";
import { THEME_META } from "../../lib/theme";

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("Settings ▸ Look theme cards", () => {
  it("paints each card's swatch band from that theme's own map", () => {
    const wrapper = mount(AppearanceCard);
    for (const meta of THEME_META) {
      const band = wrapper.get(`[data-test='theme-${meta.id}'] [data-theme]`);
      // The band wears the theme it advertises, so ui/tokens.css stays the one
      // place a hex lives — the card never carries a colour of its own.
      expect(band.attributes("data-theme"), meta.id).toBe(meta.id);
      expect(band.attributes("style"), meta.id).toBeUndefined();
    }
  });

  it("says the tone in words, not the machine value", () => {
    const wrapper = mount(AppearanceCard);
    for (const meta of THEME_META) {
      const card = wrapper.get(`[data-test='theme-${meta.id}']`);
      expect(card.text(), meta.id).toContain(meta.toneLabel);
      // The bare ThemeTone union is the machine value and never reaches a card.
      expect(card.text(), meta.id).not.toMatch(/(^|\s)(dark|light)(\s|$)/);
    }
  });
});
