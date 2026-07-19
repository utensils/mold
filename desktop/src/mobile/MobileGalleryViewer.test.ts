import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GalleryImage } from "../lib/api/types";

const { streamableMediaUrl, evictMedia } = vi.hoisted(() => ({
  streamableMediaUrl: vi.fn(),
  evictMedia: vi.fn(),
}));

vi.mock("../lib/gallery/media", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/gallery/media")>()),
  streamableMediaUrl,
  evictMedia,
}));

import MobileGalleryViewer from "./MobileGalleryViewer.vue";

const target = { baseUrl: "http://studio.tailnet.ts.net:7680", apiKey: "secret" };
const image: GalleryImage = {
  filename: "print one.png",
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

let wrapper: VueWrapper | null = null;

function mountViewer(item: GalleryImage = image): VueWrapper {
  wrapper = mount(MobileGalleryViewer, {
    attachTo: document.body,
    props: {
      item,
      target,
      cacheKey: "studio",
      hostName: "Studio",
      thumbnailUrl: "blob:thumbnail",
    },
  });
  return wrapper;
}

beforeEach(() => {
  streamableMediaUrl.mockReset().mockResolvedValue("https://studio/media/full");
  evictMedia.mockReset();
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  document.body.innerHTML = "";
});

describe("MobileGalleryViewer", () => {
  it("opens full image media in an accessible viewer with explicit reuse", async () => {
    const view = mountViewer();
    await flushPromises();

    const dialog = view.get("[role='dialog']");
    expect(dialog.element.tagName).toBe("DIALOG");
    expect(dialog.attributes("aria-modal")).toBe("true");
    expect(dialog.attributes("open")).toBe("");
    expect(view.get("[data-test='gallery-viewer-image']").attributes("src")).toBe(
      "https://studio/media/full",
    );
    expect(streamableMediaUrl).toHaveBeenCalledWith("/api/gallery/image/print%20one.png", {
      target,
      cacheKey: "studio",
      allowLegacyBlob: true,
    });

    await view.get("[data-test='gallery-viewer-reuse']").trigger("click");
    expect(view.emitted("reuse")).toHaveLength(1);
  });

  it("renders MP4 gallery items with native inline playback controls", async () => {
    const view = mountViewer({
      ...image,
      filename: "developed clip.mp4",
      format: "mp4",
    });
    await flushPromises();

    const video = view.get("[data-test='gallery-viewer-video']");
    expect(video.attributes()).toMatchObject({
      src: "https://studio/media/full",
      controls: "",
      playsinline: "",
      preload: "metadata",
    });
    expect(streamableMediaUrl).toHaveBeenCalledWith("/api/gallery/image/developed%20clip.mp4", {
      target,
      cacheKey: "studio",
      allowLegacyBlob: false,
    });
    expect(view.find("[role='status']").exists()).toBe(true);
    await video.trigger("loadedmetadata");
    expect(view.find("[role='status']").exists()).toBe(false);

    await view.get("dialog").trigger("cancel");
    expect(view.emitted("close")).toHaveLength(1);
  });

  it("shows a retryable error when full media cannot be fetched", async () => {
    streamableMediaUrl.mockRejectedValueOnce(new Error("offline"));
    const view = mountViewer();
    await flushPromises();

    expect(view.get("[role='alert']").text()).toContain("Couldn’t load");
    await view.get("[role='alert'] button").trigger("click");
    await flushPromises();

    expect(streamableMediaUrl).toHaveBeenCalledTimes(2);
    expect(view.find("[role='alert']").exists()).toBe(false);
    expect(view.find("[data-test='gallery-viewer-image']").exists()).toBe(true);
  });

  it("preserves the host-upgrade guidance for legacy authenticated video hosts", async () => {
    streamableMediaUrl.mockRejectedValueOnce(
      new Error("Update this Mold host to stream videos on iPhone."),
    );
    const view = mountViewer({ ...image, filename: "legacy.mp4", format: "mp4" });
    await flushPromises();

    expect(view.get("[role='alert']").text()).toContain(
      "Update this Mold host to stream videos on iPhone.",
    );
  });

  it("evicts the full-media blob when the viewer closes", async () => {
    const view = mountViewer();
    await flushPromises();
    view.unmount();
    wrapper = null;

    expect(evictMedia).toHaveBeenCalledWith("/api/gallery/image/print%20one.png", "studio");
  });

  it("does not offer reuse when the host only synthesized file metadata", async () => {
    const view = mountViewer({ ...image, metadata_synthetic: true });
    await flushPromises();

    const reuse = view.get("[data-test='gallery-viewer-reuse']");
    expect(reuse.attributes("disabled")).toBe("");
    expect(reuse.text()).toBe("Prompt unavailable");
  });
});
