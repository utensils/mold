import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GalleryImage } from "../lib/api/types";

const { streamableMediaUrl, evictMedia } = vi.hoisted(() => ({
  streamableMediaUrl: vi.fn(),
  evictMedia: vi.fn(),
}));
const { invoke, apiFetchTo, apiJsonTo } = vi.hoisted(() => ({
  invoke: vi.fn(),
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(),
}));

vi.mock("../lib/gallery/media", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/gallery/media")>()),
  streamableMediaUrl,
  evictMedia,
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke }));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiFetchTo,
  apiJsonTo,
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

function mountViewer(
  item: GalleryImage = image,
  navigation: {
    position?: number;
    total?: number;
    hasPrevious?: boolean;
    hasNext?: boolean;
  } = {},
): VueWrapper {
  wrapper = mount(MobileGalleryViewer, {
    attachTo: document.body,
    props: {
      item,
      target,
      cacheKey: "studio",
      hostName: "Studio",
      thumbnailUrl: "blob:thumbnail",
      ...navigation,
    },
  });
  return wrapper;
}

beforeEach(() => {
  streamableMediaUrl.mockReset().mockResolvedValue("https://studio/media/full");
  evictMedia.mockReset();
  invoke.mockReset().mockResolvedValue(undefined);
  apiFetchTo.mockReset().mockResolvedValue({
    blob: () => Promise.resolve(new Blob([Uint8Array.from([1, 2, 3])])),
  } as Response);
  apiJsonTo.mockReset().mockResolvedValue({
    formats: ["gif", "apng", "webp"],
    gif_playback: ["loop", "bounce"],
    gif_repeat: ["forever", "once"],
  });
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  document.body.innerHTML = "";
});

describe("MobileGalleryViewer", () => {
  it("leaves the native iOS image context menu available", async () => {
    const view = mountViewer();
    await flushPromises();

    const contextMenu = new Event("contextmenu", { bubbles: true, cancelable: true });
    view.get("[data-test='gallery-viewer-image']").element.dispatchEvent(contextMenu);

    expect(contextMenu.defaultPrevented).toBe(false);
  });

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

  it("shows the recorded runtime pipeline for an LTX video", async () => {
    const view = mountViewer({
      ...image,
      filename: "ltx-print.mp4",
      format: "mp4",
      metadata: {
        ...image.metadata,
        model: "ltx-2.3-22b-dev:fp8",
        frames: 97,
        fps: 24,
        pipeline: "two-stage-hq",
      },
    });
    await flushPromises();

    expect(view.get("[data-test='gallery-viewer-pipeline']").text()).toContain("two-stage-hq");
  });

  it("offers a still print as a generation source when the selected model supports it", async () => {
    const view = mountViewer();
    await view.setProps({ canUseAsSource: true });
    await flushPromises();

    const source = view.get("[data-test='gallery-viewer-use-source']");
    expect(source.text()).toBe("Use as source");
    await source.trigger("click");
    expect(view.emitted("use-source")).toHaveLength(1);

    await view.setProps({ item: { ...image, filename: "clip.mp4", format: "mp4" } });
    await flushPromises();
    expect(view.find("[data-test='gallery-viewer-use-source']").exists()).toBe(false);
  });

  it("copies and saves the authenticated full-resolution still through native iOS actions", async () => {
    const view = mountViewer();
    await flushPromises();

    await view.get("[data-test='gallery-viewer-copy']").trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/image/print%20one.png");
    expect(invoke).toHaveBeenCalledWith("copy_image_to_clipboard", {
      dataB64: "AQID",
    });
    expect(view.get("[data-test='gallery-viewer-action-status']").text()).toBe("Image copied");

    await view.get("[data-test='gallery-viewer-save']").trigger("click");
    await flushPromises();
    expect(invoke).toHaveBeenCalledWith("save_image_to_photos", {
      dataB64: "AQID",
    });
    expect(view.get("[data-test='gallery-viewer-action-status']").text()).toBe("Sent to Photos");
  });

  it("uses an expanded generated-result URL without refetching gallery media", async () => {
    const view = mountViewer();
    await view.setProps({ mediaUrlOverride: "blob:generated-result" });
    await flushPromises();

    expect(view.get("[data-test='gallery-viewer-image']").attributes("src")).toBe(
      "blob:generated-result",
    );
    expect(streamableMediaUrl).toHaveBeenCalledTimes(1);
  });

  it("marks upscaled stills with the shared print badge", async () => {
    const view = mountViewer({
      ...image,
      filename: "print one-upscaled.png",
      metadata: {
        ...image.metadata,
        upscale_model: "real-esrgan-x4plus",
        generation_width: 512,
        generation_height: 512,
      },
    });
    await flushPromises();

    expect(view.get("[data-test='upscaled-badge']").text()).toBe("Upscaled");
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

    await video.trigger("pointerdown", {
      pointerId: 7,
      pointerType: "touch",
      isPrimary: true,
      clientX: 280,
      clientY: 200,
    });
    await view.get("[data-test='gallery-viewer-stage']").trigger("pointerup", {
      pointerId: 7,
      pointerType: "touch",
      isPrimary: true,
      clientX: 80,
      clientY: 200,
    });
    expect(view.emitted("next")).toBeUndefined();

    await view.get("dialog").trigger("cancel");
    expect(view.emitted("close")).toHaveLength(1);
  });

  it("saves the original MP4 to Photos through the native iOS bridge", async () => {
    streamableMediaUrl
      .mockResolvedValueOnce("https://studio/media/playback-ticket")
      .mockResolvedValueOnce("https://studio/media/save-ticket");
    const view = mountViewer({
      ...image,
      filename: "developed clip.mp4",
      format: "mp4",
    });
    await flushPromises();

    await view.get("[data-test='gallery-viewer-save-video']").trigger("click");
    await flushPromises();

    expect(streamableMediaUrl).toHaveBeenLastCalledWith("/api/gallery/image/developed%20clip.mp4", {
      target,
      cacheKey: "studio",
      allowLegacyBlob: false,
    });
    expect(invoke).toHaveBeenCalledWith("save_video_to_photos", {
      url: "https://studio/media/save-ticket",
    });
    expect(view.get("[data-test='gallery-viewer-action-status']").text()).toBe("Saved to Photos");
  });

  it("reports native video save failures without closing the viewer", async () => {
    invoke.mockRejectedValueOnce(new Error("Photos access is denied"));
    const view = mountViewer({ ...image, filename: "clip.mp4", format: "mp4" });
    await flushPromises();

    await view.get("[data-test='gallery-viewer-save-video']").trigger("click");
    await flushPromises();

    expect(view.find("[role='dialog']").exists()).toBe(true);
    expect(view.get("[data-test='gallery-viewer-action-status']").text()).toBe(
      "Photos access is denied",
    );
  });

  it("opens video export options for an MP4", async () => {
    const view = mountViewer({ ...image, filename: "developed clip.mp4", format: "mp4" });
    await flushPromises();

    await view.get("[data-test='gallery-viewer-export']").trigger("click");
    await flushPromises();
    expect(apiJsonTo).toHaveBeenCalledWith(target, "/api/gallery/export-options");
    expect(view.get("[data-test='video-export-dialog']").text()).toContain("Bounce");
    expect(view.get("[data-test='video-export-dialog']").text()).toContain("WEBP");
  });

  it("fails closed when the generated video's exact host is unavailable", async () => {
    wrapper = mount(MobileGalleryViewer, {
      attachTo: document.body,
      props: {
        item: { ...image, filename: "remote-result.mp4", format: "mp4" },
        target: { baseUrl: "", apiKey: null },
        cacheKey: "missing-origin",
        hostName: "Unavailable host",
        thumbnailUrl: "blob:generated-video",
        mediaUrlOverride: "blob:generated-video",
        exportEnabled: false,
      },
    });
    await flushPromises();

    expect(wrapper.find("[data-test='gallery-viewer-export']").exists()).toBe(false);
    expect(wrapper.find("[data-test='gallery-viewer-save-video']").exists()).toBe(false);
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

  it("reloads changed image items and evicts each item's own media path", async () => {
    streamableMediaUrl
      .mockResolvedValueOnce("https://studio/media/one")
      .mockResolvedValueOnce("https://studio/media/two");
    const view = mountViewer();
    await flushPromises();

    expect(view.get("[data-test='gallery-viewer-image']").attributes("src")).toBe(
      "https://studio/media/one",
    );

    await view.setProps({
      item: { ...image, filename: "print two.png" },
      thumbnailUrl: "blob:thumbnail-two",
    });
    await flushPromises();

    expect(streamableMediaUrl).toHaveBeenLastCalledWith("/api/gallery/image/print%20two.png", {
      target,
      cacheKey: "studio",
      allowLegacyBlob: true,
    });
    expect(view.get("[data-test='gallery-viewer-image']").attributes("src")).toBe(
      "https://studio/media/two",
    );
    expect(evictMedia).toHaveBeenCalledWith("/api/gallery/image/print%20one.png", "studio");

    view.unmount();
    wrapper = null;
    expect(evictMedia).toHaveBeenLastCalledWith("/api/gallery/image/print%20two.png", "studio");
  });

  it("reloads across hosts and ignores a stale resolution from the previous host", async () => {
    let resolveStudio!: (url: string) => void;
    streamableMediaUrl
      .mockImplementationOnce(
        () =>
          new Promise<string>((resolve) => {
            resolveStudio = resolve;
          }),
      )
      .mockResolvedValueOnce("https://remote/media/full");
    const view = mountViewer();
    const remoteTarget = { baseUrl: "https://remote.example.com", apiKey: "remote-secret" };

    await view.setProps({
      target: remoteTarget,
      cacheKey: "remote",
      hostName: "Remote",
    });
    await flushPromises();

    expect(streamableMediaUrl).toHaveBeenLastCalledWith("/api/gallery/image/print%20one.png", {
      target: remoteTarget,
      cacheKey: "remote",
      allowLegacyBlob: true,
    });
    expect(view.get("[data-test='gallery-viewer-image']").attributes("src")).toBe(
      "https://remote/media/full",
    );
    expect(evictMedia).toHaveBeenCalledWith("/api/gallery/image/print%20one.png", "studio");

    resolveStudio("https://studio/media/stale");
    await flushPromises();
    expect(view.get("[data-test='gallery-viewer-image']").attributes("src")).toBe(
      "https://remote/media/full",
    );

    view.unmount();
    wrapper = null;
    expect(evictMedia).toHaveBeenLastCalledWith("/api/gallery/image/print%20one.png", "remote");
  });

  it("does not offer reuse when the host only synthesized file metadata", async () => {
    const view = mountViewer({ ...image, metadata_synthetic: true });
    await flushPromises();

    const reuse = view.get("[data-test='gallery-viewer-reuse']");
    expect(reuse.attributes("disabled")).toBe("");
    expect(reuse.text()).toBe("Prompt unavailable");
  });

  it("shows prepared sibling position and source prompt with graceful legacy absence", async () => {
    const prepared = {
      ...image,
      metadata: {
        ...image.metadata,
        original_prompt: "a lighthouse",
        batch_id: "batch-1",
        batch_index: 2,
        batch_count: 3,
      },
    };
    const view = mountViewer(prepared);
    await flushPromises();
    expect(view.get('[data-test="gallery-viewer-batch"]').text()).toBe("Batch 2 of 3");
    expect(view.get('[data-test="gallery-viewer-original-prompt"]').text()).toContain(
      "a lighthouse",
    );

    await view.setProps({ item: image });
    expect(view.find('[data-test="gallery-viewer-batch"]').exists()).toBe(false);
    expect(view.find('[data-test="gallery-viewer-original-prompt"]').exists()).toBe(false);
  });

  it("shows an accessible position and previous/next controls", async () => {
    const view = mountViewer(image, { position: 2, total: 4 });
    await flushPromises();

    expect(view.get("[data-test='gallery-viewer-position']").text()).toBe("2 of 4");
    expect(view.get("[data-test='gallery-viewer-position']").attributes()).toMatchObject({
      role: "status",
      "aria-live": "polite",
    });

    const previous = view.get("[data-test='gallery-viewer-previous']");
    const next = view.get("[data-test='gallery-viewer-next']");
    expect(previous.attributes("aria-label")).toBe("Previous print");
    expect(next.attributes("aria-label")).toBe("Next print");

    await previous.trigger("click");
    await next.trigger("click");
    expect(view.emitted("previous")).toHaveLength(1);
    expect(view.emitted("next")).toHaveLength(1);
  });

  it("disables navigation at the gallery boundaries", async () => {
    const view = mountViewer(image, {
      position: 1,
      total: 3,
      hasPrevious: false,
      hasNext: true,
    });
    await flushPromises();

    const previous = view.get("[data-test='gallery-viewer-previous']");
    const next = view.get("[data-test='gallery-viewer-next']");
    expect(previous.attributes("disabled")).toBe("");
    expect(next.attributes("disabled")).toBeUndefined();
    await previous.trigger("click");
    await next.trigger("click");
    expect(view.emitted("previous")).toBeUndefined();
    expect(view.emitted("next")).toHaveLength(1);
  });

  it("navigates horizontally with swipe gestures and ignores vertical drags", async () => {
    const view = mountViewer(image, { position: 2, total: 4 });
    await flushPromises();
    const stage = view.get("[data-test='gallery-viewer-stage']");

    await stage.trigger("pointerdown", {
      pointerId: 11,
      pointerType: "touch",
      isPrimary: true,
      clientX: 300,
      clientY: 200,
    });
    await stage.trigger("pointerup", {
      pointerId: 11,
      pointerType: "touch",
      isPrimary: true,
      clientX: 180,
      clientY: 208,
    });
    expect(view.emitted("next")).toHaveLength(1);

    await stage.trigger("pointerdown", {
      pointerId: 12,
      pointerType: "touch",
      isPrimary: true,
      clientX: 80,
      clientY: 200,
    });
    await stage.trigger("pointerup", {
      pointerId: 12,
      pointerType: "touch",
      isPrimary: true,
      clientX: 190,
      clientY: 206,
    });
    expect(view.emitted("previous")).toHaveLength(1);

    await stage.trigger("pointerdown", {
      pointerId: 13,
      pointerType: "touch",
      isPrimary: true,
      clientX: 180,
      clientY: 100,
    });
    await stage.trigger("pointerup", {
      pointerId: 13,
      pointerType: "touch",
      isPrimary: true,
      clientX: 170,
      clientY: 220,
    });
    expect(view.emitted("previous")).toHaveLength(1);
    expect(view.emitted("next")).toHaveLength(1);
  });

  it("supports keyboard navigation without stealing keys from video controls", async () => {
    const view = mountViewer(image, { position: 2, total: 4 });
    await flushPromises();

    const dialog = view.get("dialog");
    await dialog.trigger("keydown", { key: "ArrowLeft" });
    await dialog.trigger("keydown", { key: "ArrowRight" });
    expect(view.emitted("previous")).toHaveLength(1);
    expect(view.emitted("next")).toHaveLength(1);
  });
});
