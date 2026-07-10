import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DOMWrapper, flushPromises, mount } from "@vue/test-utils";
import { nextTick } from "vue";
import ImagePickerModal from "./ImagePickerModal.vue";
import type { GalleryImage } from "../../lib/api/types";

const galleryItems: GalleryImage[] = [
  {
    filename: "a.png",
    metadata: { prompt: "", model: "m", seed: 1, steps: 4, guidance: 3, width: 8, height: 8 },
    timestamp: 2,
  },
  {
    filename: "b.png",
    metadata: { prompt: "", model: "m", seed: 1, steps: 4, guidance: 3, width: 8, height: 8 },
    timestamp: 1,
  },
];

const apiJson = vi.fn();
const apiFetch = vi.fn();
vi.mock("../../lib/api/client", () => ({
  apiJson: (...args: unknown[]) => apiJson(...args),
  apiFetch: (...args: unknown[]) => apiFetch(...args),
}));

// AuthedMedia resolves thumbnails through authedMediaUrl; stub it so mounting
// the gallery grid doesn't hit the network.
vi.mock("../../lib/gallery/media", async () => {
  const actual =
    await vi.importActual<typeof import("../../lib/gallery/media")>("../../lib/gallery/media");
  return { ...actual, authedMediaUrl: vi.fn(() => Promise.resolve("blob:thumb")) };
});

describe("ImagePickerModal", () => {
  function bodyGet<T extends Element>(selector: string): DOMWrapper<T> {
    const element = document.body.querySelector<T>(selector);
    expect(element).not.toBeNull();
    return new DOMWrapper(element as T);
  }

  beforeEach(() => {
    apiJson.mockReset().mockResolvedValue(galleryItems);
    apiFetch.mockReset().mockResolvedValue({
      blob: () => Promise.resolve(new Blob(["x"], { type: "image/png" })),
    });
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("lists gallery images and emits the picked image with base64", async () => {
    const wrapper = mount(ImagePickerModal, {
      props: { open: true },
      attachTo: document.body,
    });
    await flushPromises();
    await nextTick();

    await bodyGet<HTMLButtonElement>("[data-test='picker-tab-gallery']").trigger("click");
    await nextTick();

    const thumbs = document.body.querySelectorAll("[data-test='picker-gallery-item']");
    expect(thumbs.length).toBe(2);

    await new DOMWrapper(thumbs[0] as HTMLElement).trigger("click");
    // blobToBase64 resolves via a FileReader event, not a microtask.
    await vi.waitFor(() => expect(wrapper.emitted("pick")).toBeTruthy());

    const picked = wrapper.emitted("pick")?.[0]?.[0] as Array<{ filename: string; base64: string }>;
    expect(picked).toHaveLength(1);
    expect(picked[0]!.filename).toBe("a.png");
    // Blob(["x"]) base64-encodes to "eA==".
    expect(picked[0]!.base64).toBe("eA==");
    expect(wrapper.emitted("close")).toBeTruthy();
  });

  it("does not render when closed", () => {
    mount(ImagePickerModal, { props: { open: false }, attachTo: document.body });
    expect(document.body.querySelector("[data-test='picker-tab-gallery']")).toBeNull();
  });

  it("exposes a labelled modal dialog and closes on Escape", async () => {
    const wrapper = mount(ImagePickerModal, {
      props: { open: true, title: "Keyframe image" },
      attachTo: document.body,
    });
    await flushPromises();

    const dialog = bodyGet<HTMLElement>("[role='dialog']");
    expect(dialog.attributes("aria-modal")).toBe("true");
    expect(dialog.attributes("aria-label")).toBe("Keyframe image");

    await bodyGet<HTMLElement>("[data-test='picker-tab-upload']").trigger("keydown", {
      key: "Escape",
    });
    expect(wrapper.emitted("close")).toBeTruthy();
  });
});
