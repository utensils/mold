import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it } from "vitest";
import Lightbox from "./Lightbox.vue";
import type { GalleryImage } from "../../types";

const item: GalleryImage = {
  filename: "print.png",
  timestamp: 1_700_000_000,
  format: "png",
  metadata: {
    prompt: "a lighthouse at dusk",
    model: "flux-dev:fp16",
    seed: 4242,
    steps: 20,
    guidance: 3.5,
    width: 1024,
    height: 768,
    version: "test",
  },
};

function setViewportWidth(px: number) {
  Object.defineProperty(window, "innerWidth", {
    value: px,
    configurable: true,
    writable: true,
  });
}

function mountWide(props: Record<string, unknown> = {}) {
  setViewportWidth(1200);
  return mount(Lightbox, {
    props: {
      item,
      index: 1,
      total: 5,
      hasPrev: true,
      hasNext: true,
      muted: true,
      ...props,
    },
    global: { stubs: { Transition: false } },
  });
}

describe("Lightbox (desktop two-pane)", () => {
  beforeEach(() => {
    setViewportWidth(1200);
    document.body.style.overflow = "";
  });

  it("renders the print details panel", () => {
    const wrapper = mountWide();
    const text = wrapper.text();
    expect(text).toContain("Print details");
    expect(text).toContain("2 / 5");
    expect(text).toContain("a lighthouse at dusk");
    expect(text).toContain("flux-dev");
    expect(text).toContain("4242");
    expect(text).toContain("1024");
    expect(text).toContain("Reuse these settings");
    expect(text).toContain("Use as source");
    expect(text).toContain("Download");
  });

  it("navigates with arrow keys and closes on Escape", async () => {
    const wrapper = mountWide();
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("next")).toHaveLength(1);
    expect(wrapper.emitted("prev")).toHaveLength(1);
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("emits prev/next from the circle buttons", async () => {
    const wrapper = mountWide();
    await wrapper.find(".lb__nav--prev").trigger("click");
    await wrapper.find(".lb__nav--next").trigger("click");
    expect(wrapper.emitted("prev")).toHaveLength(1);
    expect(wrapper.emitted("next")).toHaveLength(1);
  });

  it("emits reuse and use-source with the item", async () => {
    const wrapper = mountWide();
    await wrapper.find(".lb__reuse").trigger("click");
    await wrapper.findAll(".lb__quiet")[0]!.trigger("click");
    expect((wrapper.emitted("reuse")![0]![0] as GalleryImage).filename).toBe(
      "print.png",
    );
    expect(
      (wrapper.emitted("use-source")![0]![0] as GalleryImage).filename,
    ).toBe("print.png");
  });

  it("exposes Upscale and Delete via the overflow menu", async () => {
    const wrapper = mountWide();
    expect(wrapper.find(".lb__menu").exists()).toBe(false);
    await wrapper.find(".lb__morebtn").trigger("click");
    const items = wrapper.findAll(".lb__menu button");
    expect(items.map((b) => b.text())).toEqual(["Upscale…", "Delete"]);
    await items[1]!.trigger("click");
    expect((wrapper.emitted("delete")![0]![0] as GalleryImage).filename).toBe(
      "print.png",
    );
  });

  it("disables navigation at the ends", () => {
    const wrapper = mountWide({ hasPrev: false, hasNext: false });
    expect(
      (wrapper.find(".lb__nav--prev").element as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (wrapper.find(".lb__nav--next").element as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});

describe("Lightbox (mobile full-screen)", () => {
  it("renders the full-screen viewer with metadata chips", () => {
    setViewportWidth(480);
    const wrapper = mount(Lightbox, {
      props: {
        item,
        index: 0,
        total: 3,
        hasPrev: false,
        hasNext: true,
        muted: true,
      },
      global: { stubs: { Transition: false } },
    });
    expect(wrapper.find(".lb__full").exists()).toBe(true);
    expect(wrapper.find(".lb__card").exists()).toBe(false);
    const text = wrapper.text();
    expect(text).toContain("a lighthouse at dusk");
    expect(text).toContain("seed 4242");
    expect(text).toContain("Reuse these settings");
  });
});
