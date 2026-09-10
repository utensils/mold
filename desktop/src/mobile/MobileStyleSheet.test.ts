import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { resetOverlayStackForTests } from "@ui/lib/overlayStack";
import MobileStyleSheet from "./MobileStyleSheet.vue";

interface Row {
  name: string;
  family: string;
  description?: string | null;
}

const models: Row[] = [
  { name: "z-image-turbo:q8", family: "zimage", description: "Z-Image Turbo" },
  { name: "flux-dev:q4", family: "flux", description: "FLUX Dev" },
];

function mountSheet(props: Record<string, unknown> = {}) {
  return mount(MobileStyleSheet, {
    props: {
      open: true,
      models,
      selected: models[0]!,
      kicker: "still picture styles",
      emptyLabel: "No still picture styles on this machine.",
      ...props,
    },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  resetOverlayStackForTests();
});

describe("mobile style sheet chrome", () => {
  it("is a grabbed, scrimmed bottom sheet with a text Done in its header", async () => {
    const wrapper = mountSheet();
    await flushPromises();

    expect(wrapper.attributes("aria-modal")).toBe("true");
    expect(wrapper.attributes("data-test")).toBe("mobile-style-sheet");
    expect(wrapper.find(".mobile-sheet-grabber").exists()).toBe(true);
    const done = wrapper.get("[data-test='mobile-style-sheet-done']");
    // A text control in the header, never the 44px circular glyph the More
    // settings surface used to carry.
    expect(done.text()).toBe("Done");
    expect(done.element.closest(".mobile-sheet-head")).not.toBeNull();
    expect(wrapper.get(".mobile-sheet-title").text()).toBe("Style");
    expect(wrapper.get(".mobile-sheet-kicker").text()).toBe("still picture styles");

    await done.trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
    await wrapper.get("[data-test='mobile-style-sheet-scrim']").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(2);
    wrapper.unmount();
  });

  it("hosts the shared style menu with finger-sized rows and picks from it", async () => {
    const wrapper = mountSheet();
    await flushPromises();

    const menu = wrapper.get("[data-test='model-picker-menu']");
    expect(menu.classes()).toContain("ms-model__menu--touch");
    const ids = wrapper.findAll("[data-test='model-option-id']").map((row) => row.text());
    expect(ids).toEqual([models[0]!.name, models[1]!.name]);

    await wrapper.findAll("[data-test='model-option-id']")[1]!.trigger("click");
    expect(wrapper.emitted("pick")?.[0]?.[0]).toMatchObject({ name: models[1]!.name });

    await wrapper.get("[data-test='browse-catalog']").trigger("click");
    expect(wrapper.emitted("browse")).toHaveLength(1);
    wrapper.unmount();
  });

  it("keeps the style a machine no longer has as its own row", async () => {
    const wrapper = mountSheet({ selected: null, missingModel: "wan22-i2v-a14b:q8" });
    await flushPromises();

    const phantom = wrapper.get("[data-test='model-option-missing']");
    expect(phantom.text()).toContain("Not on this machine");
    await phantom.trigger("click");
    expect(wrapper.emitted("pick-missing")?.[0]?.[0]).toBe("wan22-i2v-a14b:q8");
    wrapper.unmount();
  });

  it("registers on the overlay stack so only the top sheet takes Escape", async () => {
    const lower = mountSheet();
    const upper = mountSheet();
    await flushPromises();

    await lower.trigger("keydown", { key: "Escape" });
    expect(lower.emitted("close")).toBeUndefined();
    await upper.trigger("keydown", { key: "Escape" });
    expect(upper.emitted("close")).toHaveLength(1);
    upper.unmount();
    lower.unmount();
  });

  it("mounts the list only while open, so every opening starts at the top", async () => {
    const wrapper = mountSheet({ open: false });
    await flushPromises();

    expect(wrapper.attributes("inert")).toBeDefined();
    expect(wrapper.find("[data-test='model-picker-menu']").exists()).toBe(false);
    await wrapper.setProps({ open: true });
    await flushPromises();
    expect(wrapper.find("[data-test='model-picker-menu']").exists()).toBe(true);
    wrapper.unmount();
  });
});
