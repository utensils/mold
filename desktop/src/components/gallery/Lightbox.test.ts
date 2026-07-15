import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../../lib/ipc", () => ({
  ipc: { getOutputDir: vi.fn().mockResolvedValue(null), revealOutputFile: vi.fn() },
}));

import Lightbox from "./Lightbox.vue";
import { useContextMenuStore } from "../../stores/contextMenu";
import { useComposerStore } from "../../stores/composer";
import type { GalleryImage } from "../../lib/api/types";

const item: GalleryImage = {
  filename: "print-0001.png",
  timestamp: 1_700_000_000,
  format: "png",
  metadata: {
    prompt: "a lighthouse at dusk",
    model: "flux-dev:q8",
    seed: 42,
    steps: 4,
    guidance: 3.5,
    width: 1024,
    height: 1024,
  },
};

beforeEach(() => {
  setActivePinia(createPinia());
});

function mountLightbox(selectedItem: GalleryImage = item) {
  return mount(Lightbox, {
    props: { item: selectedItem, index: 0, count: 3, video: false },
    global: { stubs: { AuthedMedia: { template: "<div />" } } },
  });
}

describe("Lightbox reuse", () => {
  it("restores generation dimensions instead of the upscaled raster", async () => {
    const wrapper = mountLightbox({
      ...item,
      metadata: {
        ...item.metadata,
        width: 4096,
        height: 4096,
        generation_width: 1024,
        generation_height: 1024,
        upscale_model: "real-esrgan-x4plus:fp16",
      },
    });

    await wrapper.get("button.bg-safelight").trigger("click");

    expect(useComposerStore().prefill).toMatchObject({
      width: 1024,
      height: 1024,
      upscaleModel: "real-esrgan-x4plus:fp16",
    });
  });
});

describe("Lightbox a11y", () => {
  it("is a labelled modal dialog", () => {
    const wrapper = mountLightbox();
    const dialog = wrapper.get("[role='dialog']");
    expect(dialog.attributes("aria-modal")).toBe("true");
    expect(dialog.attributes("aria-label")).toBe("Print 1 of 3");
  });

  it("labels the close and navigation controls", () => {
    const wrapper = mountLightbox();
    expect(wrapper.find("[aria-label='Close']").exists()).toBe(true);
    expect(wrapper.find("[aria-label='Previous print']").exists()).toBe(true);
    expect(wrapper.find("[aria-label='Next print']").exists()).toBe(true);
  });

  it("offers full image copy from the still-image context menu", async () => {
    const wrapper = mountLightbox();

    await wrapper.get('[data-test="lightbox-media"]').trigger("contextmenu");

    const menu = useContextMenuStore();
    expect(menu.visible).toBe(true);
    expect(menu.entries).toEqual(
      expect.arrayContaining([expect.objectContaining({ label: "Copy image", disabled: false })]),
    );
    wrapper.unmount();
  });
});
