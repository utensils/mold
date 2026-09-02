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
const { isNativeAndroidRuntime, isNativeIOSRuntime } = vi.hoisted(() => ({
  isNativeAndroidRuntime: vi.fn(),
  isNativeIOSRuntime: vi.fn(),
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
vi.mock("./platform", () => ({ isNativeAndroidRuntime, isNativeIOSRuntime }));

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
  isNativeAndroidRuntime.mockReset().mockReturnValue(false);
  isNativeIOSRuntime.mockReset().mockReturnValue(true);
  streamableMediaUrl.mockReset().mockResolvedValue("https://studio/media/full");
  evictMedia.mockReset();
  invoke.mockReset().mockResolvedValue(undefined);
  apiFetchTo.mockReset().mockResolvedValue({
    blob: () => Promise.resolve(new Blob([Uint8Array.from([1, 2, 3])], { type: "image/gif" })),
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
  Reflect.deleteProperty(navigator, "share");
  Reflect.deleteProperty(navigator, "canShare");
  Reflect.deleteProperty(URL, "createObjectURL");
  Reflect.deleteProperty(URL, "revokeObjectURL");
  vi.restoreAllMocks();
});

describe("MobileGalleryViewer", () => {
  it("makes the prompt selectable and copies it in the viewer and Info sheet", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    const view = mountViewer();
    await flushPromises();

    const viewerPrompt = view.get(".gallery-viewer-prompt > p[data-selectable]");
    expect(viewerPrompt.text()).toBe("a lighthouse at dusk");
    await view.get("[data-test='gallery-viewer-copy-prompt']").trigger("click");
    await flushPromises();
    expect(view.get("[data-test='gallery-viewer-copy-status']").text()).toBe("Prompt copied");

    await view.get("[data-test='gallery-viewer-info']").trigger("click");
    const infoPrompt = view.get("[data-test='gallery-viewer-info-prompt']");
    expect(infoPrompt.attributes()).toHaveProperty("data-selectable");
    await view.get("[data-test='gallery-viewer-info-copy-prompt']").trigger("click");
    await flushPromises();

    expect(writeText).toHaveBeenNthCalledWith(1, "a lighthouse at dusk");
    expect(writeText).toHaveBeenNthCalledWith(2, "a lighthouse at dusk");
    expect(view.get("[data-test='gallery-viewer-info-copy-status']").text()).toBe("Prompt copied");
  });

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

  it("offers upscale for both stills and videos from the full-media viewer", async () => {
    const view = mountViewer();
    await flushPromises();

    const stillUpscale = view.get("[data-test='gallery-viewer-upscale']");
    expect(stillUpscale.text()).toBe("Upscale…");
    await stillUpscale.trigger("click");
    expect(view.emitted("upscale")).toHaveLength(1);

    await view.setProps({ item: { ...image, filename: "clip.mp4", format: "mp4" } });
    await flushPromises();
    const videoUpscale = view.get("[data-test='gallery-viewer-upscale']");
    expect(videoUpscale.text()).toBe("Framewise upscale…");
    await videoUpscale.trigger("click");
    expect(view.emitted("upscale")).toHaveLength(2);
  });

  it("always allows dismissal while settings are loading", async () => {
    const view = mountViewer();
    await view.setProps({ reusing: true });

    const close = view.get("[data-test='gallery-viewer-close']");
    expect(close.attributes("disabled")).toBeUndefined();
    await close.trigger("click");
    expect(view.emitted("close")).toHaveLength(1);

    await view.get("dialog").trigger("cancel");
    expect(view.emitted("close")).toHaveLength(2);
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

    await view.get("dialog").trigger("cancel");
    expect(view.emitted("close")).toHaveLength(1);
  });

  it("swipes naturally in both directions from native video playback", async () => {
    const view = mountViewer(
      {
        ...image,
        filename: "developed clip.mp4",
        format: "mp4",
      },
      { position: 2, total: 4 },
    );
    await flushPromises();

    const video = view.get("[data-test='gallery-viewer-video']");
    await video.trigger("pointerdown", {
      pointerId: 7,
      pointerType: "touch",
      isPrimary: true,
      clientX: 280,
      clientY: 200,
    });
    window.dispatchEvent(
      new PointerEvent("pointermove", {
        pointerId: 7,
        pointerType: "touch",
        isPrimary: true,
        clientX: 180,
        clientY: 204,
      }),
    );
    window.dispatchEvent(
      new PointerEvent("pointerup", {
        pointerId: 7,
        pointerType: "touch",
        isPrimary: true,
        clientX: 80,
        clientY: 204,
      }),
    );
    expect(view.emitted("next")).toHaveLength(1);

    await video.trigger("pointerdown", {
      pointerId: 8,
      pointerType: "touch",
      isPrimary: true,
      clientX: 80,
      clientY: 200,
    });
    window.dispatchEvent(
      new PointerEvent("pointerup", {
        pointerId: 8,
        pointerType: "touch",
        isPrimary: true,
        clientX: 280,
        clientY: 204,
      }),
    );
    expect(view.emitted("previous")).toHaveLength(1);

    await video.trigger("pointerdown", {
      pointerId: 9,
      pointerType: "touch",
      isPrimary: true,
      clientX: 180,
      clientY: 200,
    });
    window.dispatchEvent(
      new PointerEvent("pointerup", {
        pointerId: 9,
        pointerType: "touch",
        isPrimary: true,
        clientX: 184,
        clientY: 202,
      }),
    );
    expect(view.emitted("next")).toHaveLength(1);
    expect(view.emitted("previous")).toHaveLength(1);
  });

  it("leaves taps and native video scrubber gestures with the playback controls", async () => {
    const view = mountViewer(
      {
        ...image,
        filename: "developed clip.mp4",
        format: "mp4",
      },
      { position: 2, total: 4 },
    );
    await flushPromises();

    const video = view.get("[data-test='gallery-viewer-video']");
    const stage = view.get("[data-test='gallery-viewer-stage']");
    vi.spyOn(video.element, "getBoundingClientRect").mockReturnValue({
      top: 0,
      bottom: 400,
      left: 0,
      right: 320,
      width: 320,
      height: 400,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });

    await video.trigger("pointerdown", {
      pointerId: 10,
      pointerType: "touch",
      isPrimary: true,
      clientX: 180,
      clientY: 200,
    });
    await stage.trigger("pointerup", {
      pointerId: 10,
      pointerType: "touch",
      isPrimary: true,
      clientX: 184,
      clientY: 202,
    });
    await video.trigger("pointerdown", {
      pointerId: 11,
      pointerType: "touch",
      isPrimary: true,
      clientX: 280,
      clientY: 380,
    });
    await stage.trigger("pointermove", {
      pointerId: 11,
      pointerType: "touch",
      isPrimary: true,
      clientX: 180,
      clientY: 380,
    });
    await stage.trigger("pointerup", {
      pointerId: 11,
      pointerType: "touch",
      isPrimary: true,
      clientX: 80,
      clientY: 380,
    });
    expect(view.emitted("next")).toBeUndefined();
    expect(view.emitted("previous")).toBeUndefined();
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

  it("opens the native iOS share sheet after the remote export completes", async () => {
    invoke.mockResolvedValueOnce("shared");
    const view = mountViewer({ ...image, filename: "developed clip.mp4", format: "mp4" });
    await flushPromises();

    await view.get("[data-test='gallery-viewer-export']").trigger("click");
    await flushPromises();
    await view.get("[data-test='video-export-dialog'] form").trigger("submit");
    await flushPromises();

    expect(invoke).toHaveBeenCalledWith("share_exported_animation", {
      url: "http://studio.tailnet.ts.net:7680/api/gallery/export/developed%20clip.mp4",
      apiKey: "secret",
      request: {
        format: "gif",
        playback: "loop",
        repeat: "forever",
        max_dimension: 720,
        fps: 12,
      },
      filename: "developed clip.gif",
      reuseKey:
        'http://studio.tailnet.ts.net:7680\ndeveloped clip.mp4\n{"format":"gif","playback":"loop","repeat":"forever","max_dimension":720,"fps":12}',
    });
    expect(apiFetchTo).not.toHaveBeenCalled();
    expect(view.get("[data-test='gallery-viewer-action-status']").text()).toBe(
      "Export ready to share",
    );
  });

  it("opens the native Android share sheet after the remote export completes", async () => {
    isNativeIOSRuntime.mockReturnValue(false);
    isNativeAndroidRuntime.mockReturnValue(true);
    invoke.mockResolvedValueOnce("shared");
    const view = mountViewer({ ...image, filename: "developed clip.mp4", format: "mp4" });
    await flushPromises();

    await view.get("[data-test='gallery-viewer-export']").trigger("click");
    await flushPromises();
    await view.get("[data-test='video-export-dialog'] form").trigger("submit");
    await flushPromises();

    expect(invoke).toHaveBeenCalledWith(
      "share_exported_animation",
      expect.objectContaining({ filename: "developed clip.gif" }),
    );
    expect(apiFetchTo).not.toHaveBeenCalled();
    expect(view.get("[data-test='gallery-viewer-action-status']").text()).toBe(
      "Export ready to share",
    );
  });

  it("downloads locally when the mobile UI runs outside native iOS", async () => {
    isNativeIOSRuntime.mockReturnValue(false);
    const share = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "share", { value: share, configurable: true });
    Object.defineProperty(navigator, "canShare", {
      value: vi.fn(() => true),
      configurable: true,
    });
    Object.defineProperty(URL, "createObjectURL", {
      value: vi.fn(() => "blob:video-export"),
      configurable: true,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      value: vi.fn(),
      configurable: true,
    });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const view = mountViewer({ ...image, filename: "developed clip.mp4", format: "mp4" });
    await flushPromises();

    await view.get("[data-test='gallery-viewer-export']").trigger("click");
    await flushPromises();
    await view.get("[data-test='video-export-dialog'] form").trigger("submit");
    await flushPromises();

    expect(click).toHaveBeenCalledOnce();
    expect(share).not.toHaveBeenCalled();
    expect(view.get("[data-test='gallery-viewer-action-status']").text()).toBe("Video exported");
  });

  it("keeps export options retryable when the native iOS share sheet cannot open", async () => {
    const view = mountViewer({ ...image, filename: "developed clip.mp4", format: "mp4" });
    await flushPromises();

    await view.get("[data-test='gallery-viewer-export']").trigger("click");
    await flushPromises();
    invoke.mockRejectedValueOnce(new Error("Couldn’t open the iOS share sheet"));
    await view.get("[data-test='video-export-dialog'] form").trigger("submit");
    await flushPromises();

    expect(view.get("[data-test='video-export-dialog']").isVisible()).toBe(true);
    expect(view.get("[role='alert']").text()).toContain("Couldn’t open the iOS share sheet");
  });

  it("keeps a cancelled native export staged for a retry", async () => {
    invoke.mockResolvedValueOnce("cancelled").mockResolvedValueOnce("shared");
    const view = mountViewer({ ...image, filename: "developed clip.mp4", format: "mp4" });
    await flushPromises();

    await view.get("[data-test='gallery-viewer-export']").trigger("click");
    await flushPromises();
    await view.get("[data-test='video-export-dialog'] form").trigger("submit");
    await flushPromises();

    expect(view.get("[data-test='video-export-dialog']").isVisible()).toBe(true);
    expect(view.find("[data-test='gallery-viewer-action-status']").exists()).toBe(false);

    await view.get("[data-test='video-export-dialog'] form").trigger("submit");
    await flushPromises();

    const calls = invoke.mock.calls.filter(([command]) => command === "share_exported_animation");
    expect(calls).toHaveLength(2);
    expect(calls[1]?.[1]).toEqual(calls[0]?.[1]);
    expect(apiFetchTo).not.toHaveBeenCalled();
    expect(view.find("[data-test='video-export-dialog']").exists()).toBe(false);
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
    expect(reuse.text()).toBe("Settings unavailable");
  });

  it("offers settings reuse for a promptless print and hides stale original prompt provenance", async () => {
    const view = mountViewer({
      ...image,
      metadata: {
        ...image.metadata,
        prompt: "",
        original_prompt: "stale prior prompt",
      },
    });
    await flushPromises();

    const reuse = view.get("[data-test='gallery-viewer-reuse']");
    expect(reuse.attributes("disabled")).toBeUndefined();
    expect(reuse.text()).toBe("Reuse settings");
    expect(view.text()).toContain("No prompt was used for this print.");
    expect(view.text()).not.toContain("stale prior prompt");
    await reuse.trigger("click");
    expect(view.emitted("reuse")).toHaveLength(1);
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

  /// Dragging a mesh is how you ROTATE it. The stage arms its swipe on
  /// `pointerdown.capture`, so it fires before the mesh viewer's own handler
  /// and a child cannot stop it — orbiting a model sideways navigated the
  /// gallery out from under the user instead of turning the model.
  it("rotates a mesh without navigating the gallery", async () => {
    const meshItem: GalleryImage = {
      ...image,
      filename: "chair.glb",
      format: "glb",
    };
    const view = mountViewer(meshItem, { position: 2, total: 4 });
    await flushPromises();
    const mesh = view.get("[data-test='gallery-viewer-mesh']");

    // A long horizontal drag that starts on the mesh: far past SWIPE_DISTANCE
    // and unambiguously horizontal, so nothing but the origin can save it.
    // Dispatch ON the mesh, as a finger on the model does: the stage still
    // sees it, because its listener is registered for the capture phase.
    await mesh.trigger("pointerdown", {
      pointerId: 21,
      pointerType: "touch",
      isPrimary: true,
      clientX: 300,
      clientY: 200,
    });
    await mesh.trigger("pointerup", {
      pointerId: 21,
      pointerType: "touch",
      isPrimary: true,
      clientX: 90,
      clientY: 204,
    });

    expect(view.emitted("next")).toBeUndefined();
    expect(view.emitted("previous")).toBeUndefined();
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

describe("MobileGalleryViewer info sheet", () => {
  const organization = {
    title: "Storm study",
    favorite: false,
    tags: ["Blue"],
    collections: ["portraits"],
    trashedAt: null,
    purgeAt: null,
    unresolvedCollectionIds: [],
  };
  const collections = [
    {
      slug: "portraits",
      name: "Portraits",
      count: 3,
      hostIds: ["studio"],
      hostsLabel: "Studio",
      cover: null,
      hidden: false,
    },
    {
      slug: "landscapes",
      name: "Landscapes",
      count: 1,
      hostIds: ["studio"],
      hostsLabel: "Studio",
      cover: null,
      hidden: false,
    },
  ];

  function mountOrganizedViewer(props: Record<string, unknown> = {}): VueWrapper {
    wrapper = mount(MobileGalleryViewer, {
      attachTo: document.body,
      props: {
        item: image,
        target,
        cacheKey: "studio",
        hostName: "Studio",
        thumbnailUrl: "blob:thumbnail",
        organization,
        organizeEnabled: true,
        tagSuggestions: [
          { name: "Blue", count: 4 },
          { name: "smurf", count: 2 },
        ],
        collections,
        ...props,
      },
    });
    return wrapper;
  }

  it("shows metadata Info even when the host cannot organize", async () => {
    const view = mountViewer();
    await flushPromises();
    expect(view.find("[data-test='gallery-viewer-info']").exists()).toBe(true);
    await view.get("[data-test='gallery-viewer-info']").trigger("click");
    await flushPromises();

    expect(document.activeElement).toBe(view.get(".mobile-library-sheet-panel").element);
    expect(view.get("[data-test='gallery-viewer-print-details']").text()).toContain("flux-dev:q8");
    expect(view.get("[data-test='gallery-viewer-print-details']").text()).toContain("1024×1024");
  });

  it("dismisses the Info sheet with a downward swipe from its top", async () => {
    const view = mountViewer();
    await flushPromises();
    await view.get("[data-test='gallery-viewer-info']").trigger("click");
    const panel = view.get(".mobile-library-sheet-panel").element;
    const touch = (type: string, y: number, ended = false) => {
      const event = new Event(type, { bubbles: true, cancelable: true });
      const point = { identifier: 5, clientX: 120, clientY: y };
      Object.defineProperty(event, "touches", { value: ended ? [] : [point] });
      Object.defineProperty(event, "changedTouches", { value: [point] });
      return event;
    };

    panel.dispatchEvent(touch("touchstart", 100));
    const move = touch("touchmove", 240);
    panel.dispatchEvent(move);
    expect(move.defaultPrevented).toBe(true);
    await flushPromises();
    expect(view.get(".mobile-library-sheet-panel").attributes("style")).toContain("translateY");
    panel.dispatchEvent(touch("touchend", 240, true));
    await flushPromises();

    expect(view.get("[data-test='gallery-viewer-info-sheet']").classes()).not.toContain("is-open");
  });

  it("matches the desktop print-detail metadata fields", async () => {
    const view = mountViewer({
      ...image,
      size_bytes: 1_500_000,
      metadata: {
        ...image.metadata,
        original_prompt: "a beacon",
        negative_prompt: "fog",
        batch_id: "batch-9",
        batch_index: 2,
        batch_count: 3,
        scheduler: "euler-a",
        cfg_plus: true,
        strength: 0.65,
        frames: 97,
        fps: 24,
        pipeline: "two-stage",
        loras: [{ path: "detail.safetensors", scale: 0.8 }],
        version: "0.21.0",
      },
    });
    await flushPromises();
    await view.get("[data-test='gallery-viewer-info']").trigger("click");

    const details = view.get("[data-test='gallery-viewer-print-details']").text();
    for (const value of [
      "print one.png",
      "Original a beacon",
      "Negative fog",
      "Prepared batch 2 of 3 · batch-9",
      "Seed42",
      "Steps · guidance4 · 3.5",
      "Schedulereuler-a",
      "CFG++on",
      "Denoise strength0.65",
      "Frames97 · 24 fps",
      "LoRAdetail.safetensors × 0.80",
      "File size1.5 MB",
      "FormatPNG",
      "HostStudio",
      "mold 0.21.0",
    ]) {
      expect(details).toContain(value);
    }
  });

  it("shows the saved title as the viewer title line", async () => {
    const view = mountOrganizedViewer();
    await flushPromises();
    expect(view.get("[data-test='gallery-viewer-title']").text()).toBe("Storm study");
  });

  it("falls back to the prompt for an untitled print", async () => {
    const view = mountOrganizedViewer({ organization: { ...organization, title: null } });
    await flushPromises();
    expect(view.get("[data-test='gallery-viewer-title']").text()).toBe("a lighthouse at dusk");
  });

  it("commits an edited title on Done and clears it when blanked", async () => {
    const view = mountOrganizedViewer();
    await flushPromises();
    await view.get("[data-test='gallery-viewer-info']").trigger("click");

    const input = view.get("[data-test='gallery-viewer-title-input']");
    expect((input.element as HTMLInputElement).value).toBe("Storm study");
    await input.setValue("  Grain test 01 ");
    await input.trigger("blur");
    expect(view.emitted("rename")).toEqual([["Grain test 01"]]);

    await input.setValue("   ");
    await view.get("[data-test='gallery-viewer-title-save']").trigger("submit");
    expect(view.emitted("rename")?.at(-1)).toEqual([null]);
  });

  it("does not emit a rename when the title is unchanged", async () => {
    const view = mountOrganizedViewer();
    await flushPromises();
    await view.get("[data-test='gallery-viewer-info']").trigger("click");
    await view.get("[data-test='gallery-viewer-title-input']").trigger("blur");
    expect(view.emitted("rename")).toBeUndefined();
  });

  it("toggles the favorite and edits tags from the sheet", async () => {
    const view = mountOrganizedViewer();
    await flushPromises();
    await view.get("[data-test='gallery-viewer-info']").trigger("click");

    await view.get("[data-test='gallery-viewer-favorite']").trigger("click");
    expect(view.emitted("favorite")).toEqual([[true]]);

    await view.get("[data-test='gallery-viewer-tag-remove']").trigger("click");
    expect(view.emitted("tags")).toEqual([[{ remove: ["Blue"] }]]);

    const input = view.get("[data-test='gallery-viewer-tag-input']");
    await input.setValue("  #Grain ");
    await view.get("[data-test='gallery-viewer-tag-add']").trigger("submit");
    expect(view.emitted("tags")?.at(-1)).toEqual([{ add: ["#Grain"] }]);
  });

  it("suggests merged tags the print does not already carry", async () => {
    const view = mountOrganizedViewer();
    await flushPromises();
    await view.get("[data-test='gallery-viewer-info']").trigger("click");

    const suggestions = view.get("[data-test='gallery-viewer-tag-suggestions']");
    expect(suggestions.text()).toContain("smurf");
    expect(suggestions.text()).not.toContain("Blue");
  });

  it("toggles collection membership and creates a new collection", async () => {
    const view = mountOrganizedViewer();
    await flushPromises();
    await view.get("[data-test='gallery-viewer-info']").trigger("click");

    const options = view.findAll("[data-test='gallery-viewer-collection-option']");
    expect(options[0]?.attributes("aria-checked")).toBe("true");
    await options[1]!.trigger("click");
    expect(view.emitted("collection")).toEqual([
      [{ slug: "landscapes", name: "Landscapes", member: true }],
    ]);

    await view.get("[data-test='gallery-viewer-collection-input']").setValue("Night shots");
    await view.get("[data-test='gallery-viewer-collection-create']").trigger("submit");
    expect(view.emitted("collection")?.at(-1)).toEqual([
      [{ slug: "night-shots", name: "Night shots", member: true }][0],
    ]);
  });

  it("offers Restore and a two-step Delete forever with the purge countdown", async () => {
    const purgeAt = Math.floor(Date.now() / 1000) + 3 * 86_400;
    const view = mountOrganizedViewer({
      trashed: true,
      organization: { ...organization, trashedAt: purgeAt - 30 * 86_400, purgeAt },
    });
    await flushPromises();
    await view.get("[data-test='gallery-viewer-info']").trigger("click");

    // Trashed prints swap the organize editors for recovery actions.
    expect(view.find("[data-test='gallery-viewer-title-input']").exists()).toBe(false);
    expect(view.get("[data-test='gallery-viewer-purge']").text()).toBe("Purges in 3 d");

    const deleteButton = view.get("[data-test='gallery-viewer-delete-forever']");
    await deleteButton.trigger("click");
    expect(view.emitted("delete-forever")).toBeUndefined();
    expect(view.get("[data-test='gallery-viewer-delete-prompt']").text()).toBe(
      "Delete this print forever?",
    );
    await deleteButton.trigger("click");
    expect(view.emitted("delete-forever")).toHaveLength(1);

    await view.get("[data-test='gallery-viewer-restore']").trigger("click");
    expect(view.emitted("restore")).toHaveLength(1);
  });
});
