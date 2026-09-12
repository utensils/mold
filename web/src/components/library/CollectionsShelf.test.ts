import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import CollectionsShelf from "./CollectionsShelf.vue";

vi.mock("../../composables/useThumbnailSources", () => ({
  useThumbnailSources: () => ({ srcFor: () => "" }),
}));

describe("CollectionsShelf", () => {
  it("spells the New collection keycap in the platform's own grammar", async () => {
    const { shiftShortcutLabel } = await import("../../lib/platform");
    const wrapper = mount(CollectionsShelf, {
      props: { cards: [], canCreate: true },
    });
    expect(wrapper.get(".ccard__kbd").text()).toBe(shiftShortcutLabel("N"));
  });
});
