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

function mountLightbox(
  selectedItem: GalleryImage = item,
  video = false,
  props: Record<string, unknown> = {},
) {
  return mount(Lightbox, {
    props: { item: selectedItem, index: 0, count: 3, video, ...props },
    global: { stubs: { AuthedMedia: { template: "<div />" } } },
  });
}

describe("Lightbox reuse", () => {
  it("hands the composer the full metadata so nothing is dropped", async () => {
    const metadata = {
      ...item.metadata,
      negative_prompt: "blurry",
      width: 4096,
      height: 4096,
      generation_width: 1024,
      generation_height: 1024,
      upscale_model: "real-esrgan-x4plus:fp16",
      scheduler: "ddim",
      strength: 0.6,
      loras: [{ path: "detail.safetensors", scale: 0.8 }],
    };
    const wrapper = mountLightbox({ ...item, metadata });

    await wrapper.get("button.bg-safelight").trigger("click");

    expect(useComposerStore().prefill).toEqual({ metadata });
  });

  it("makes cached editing primary and keeps duplication explicit for sequence prints", async () => {
    const wrapper = mountLightbox(item, true, {
      isSequence: true,
      canEditSequence: true,
    });

    expect(wrapper.get("[data-test='lightbox-primary-action']").text()).toContain("Edit sequence");
    expect(wrapper.get("[data-test='lightbox-duplicate-sequence']").text()).toBe(
      "Duplicate as new",
    );

    await wrapper.get("[data-test='lightbox-primary-action']").trigger("click");
    expect(wrapper.emitted("editSequence")).toHaveLength(1);
    expect(wrapper.emitted("reuseSequence")).toBeUndefined();

    await wrapper.get("[data-test='lightbox-duplicate-sequence']").trigger("click");
    expect(wrapper.emitted("reuseSequence")).toHaveLength(1);
  });

  it("labels the safe fresh-draft fallback when no durable sequence is available", async () => {
    const wrapper = mountLightbox(item, true, {
      isSequence: true,
      canEditSequence: false,
    });

    expect(wrapper.get("[data-test='lightbox-primary-action']").text()).toContain(
      "Duplicate as new",
    );
    await wrapper.get("[data-test='lightbox-primary-action']").trigger("click");
    expect(wrapper.emitted("reuseSequence")).toHaveLength(1);
  });
});

describe("Lightbox metadata panel", () => {
  const richItem: GalleryImage = {
    filename: "print-0002.mp4",
    timestamp: 1_700_000_000,
    format: "mp4",
    size_bytes: 12_500_000,
    metadata: {
      prompt: "a ship in a storm",
      negative_prompt: "calm seas",
      original_prompt: "a ship",
      batch_id: "batch-2026-07-20",
      batch_index: 2,
      batch_count: 3,
      model: "ltx2:q8",
      seed: 7,
      steps: 30,
      guidance: 3,
      width: 768,
      height: 512,
      strength: 0.6,
      scheduler: "ddim",
      cfg_plus: true,
      frames: 121,
      fps: 30,
      loras: [
        { path: "detail.safetensors", scale: 0.8 },
        { path: "grain.safetensors", scale: 0.5 },
      ],
      output_format: "mp4",
      version: "0.17.1",
    },
  };

  it("renders the full embedded field set when present", () => {
    const wrapper = mountLightbox(richItem);
    const text = wrapper.get("aside").text();

    expect(wrapper.get("[data-test='lightbox-negative']").text()).toContain("calm seas");
    expect(wrapper.get("[data-test='lightbox-original']").text()).toContain("a ship");
    expect(wrapper.get("[data-test='lightbox-batch']").text()).toContain("2 of 3");
    expect(wrapper.get("[data-test='lightbox-batch']").text()).toContain("batch-2026-07-20");
    expect(wrapper.get("[data-test='lightbox-scheduler']").text()).toContain("ddim");
    expect(wrapper.get("[data-test='lightbox-cfg-plus']").text()).toContain("on");
    expect(wrapper.get("[data-test='lightbox-strength']").text()).toContain("0.60");
    expect(wrapper.get("[data-test='lightbox-video']").text()).toContain("121");
    expect(wrapper.get("[data-test='lightbox-video']").text()).toContain("30 fps");
    expect(wrapper.get("[data-test='lightbox-file-size']").text()).toContain("12.5 MB");
    expect(wrapper.get("[data-test='lightbox-format']").text()).toContain("MP4");
    const loras = wrapper.findAll("[data-test='lightbox-lora']");
    expect(loras).toHaveLength(2);
    expect(loras[0]!.text()).toContain("detail.safetensors");
    expect(loras[0]!.text()).toContain("0.80");
    expect(text).toContain("mold 0.17.1");
  });

  it("omits every conditional row when its field is absent", () => {
    const wrapper = mountLightbox({ ...item, format: null, size_bytes: null });
    for (const row of [
      "lightbox-negative",
      "lightbox-original",
      "lightbox-batch",
      "lightbox-scheduler",
      "lightbox-cfg-plus",
      "lightbox-strength",
      "lightbox-video",
      "lightbox-format",
      "lightbox-lora",
      "lightbox-version",
    ]) {
      expect(wrapper.find(`[data-test='${row}']`).exists()).toBe(false);
    }
  });

  it("shows a legacy single lora/lora_scale pair as a one-row stack", () => {
    const wrapper = mountLightbox({
      ...item,
      metadata: { ...item.metadata, lora: "old.safetensors", lora_scale: 0.7 },
    });
    const loras = wrapper.findAll("[data-test='lightbox-lora']");
    expect(loras).toHaveLength(1);
    expect(loras[0]!.text()).toContain("old.safetensors");
    expect(loras[0]!.text()).toContain("0.70");
  });

  it("identifies an upscaled print using its persisted generation dimensions", () => {
    const wrapper = mountLightbox({
      ...item,
      filename: "renamed-output.png",
      metadata: {
        ...item.metadata,
        width: 4096,
        height: 4096,
        generation_width: 1024,
        generation_height: 1024,
        upscale_model: "real-esrgan-x4plus",
      },
    });

    expect(wrapper.get("[data-test='upscaled-badge']").text()).toBe("Upscaled");
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
      expect.arrayContaining([
        expect.objectContaining({ label: "Copy image", disabled: false }),
        expect.objectContaining({ label: "Copy file path" }),
      ]),
    );
    wrapper.unmount();
  });
});

describe("Lightbox save action", () => {
  it("uses native save language for images and videos", () => {
    expect(mountLightbox().get("[data-test='save-media']").text()).toBe("Save image");
    const video: GalleryImage = { ...item, filename: "print-0001.mp4", format: "mp4" };
    expect(mountLightbox(video, true).get("[data-test='save-media']").text()).toBe("Save video");
  });
});
