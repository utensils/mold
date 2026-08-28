import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import AuthedMedia from "./AuthedMedia.vue";
import { authedMediaUrl, fullSizeMediaUrl } from "../../lib/gallery/media";

vi.mock("../../lib/gallery/media", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/gallery/media")>()),
  authedMediaUrl: vi.fn().mockResolvedValue("blob:video"),
  fullSizeMediaUrl: vi.fn().mockResolvedValue("http://remote/media?ticket=one-use"),
  // Outside Tauri the native cache does not apply; the blob route answers.
  prepareNativeThumbnail: vi.fn().mockResolvedValue(null),
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(authedMediaUrl).mockResolvedValue("blob:video");
  vi.mocked(fullSizeMediaUrl).mockResolvedValue("http://remote/media?ticket=one-use");
});

afterEach(() => vi.useRealTimers());

describe("AuthedMedia", () => {
  it.each([
    { video: false, filename: "remote.png", allowLegacyBlob: true, element: "img" },
    { video: true, filename: "remote.mp4", allowLegacyBlob: false, element: "video" },
  ])(
    "opens a remote-only $filename through a streamable media ticket",
    async ({ video, filename, allowLegacyBlob, element }) => {
      const target = { baseUrl: "http://plato:7680", apiKey: "secret" };
      const wrapper = mount(AuthedMedia, {
        props: {
          path: `/api/gallery/image/${filename}`,
          target,
          cacheKey: "plato-7680",
          video,
          controls: video,
        },
      });
      await vi.waitFor(() => expect(wrapper.find(element).exists()).toBe(true));

      expect(fullSizeMediaUrl).toHaveBeenCalledWith(`/api/gallery/image/${filename}`, {
        target,
        cacheKey: "plato-7680",
        allowLegacyBlob,
        video,
      });
      expect(authedMediaUrl).not.toHaveBeenCalled();
      expect(wrapper.get(element).attributes("src")).toBe("http://remote/media?ticket=one-use");
    },
  );

  it("disables Picture-in-Picture for desktop video playback", async () => {
    const wrapper = mount(AuthedMedia, {
      props: { path: "/api/gallery/video/demo.mp4", video: true, controls: true },
    });
    await vi.waitFor(() => expect(wrapper.find("video").exists()).toBe(true));

    expect(wrapper.get("video").attributes()).toHaveProperty("disablepictureinpicture");
  });

  it("retries transient failures and renders the recovered thumbnail", async () => {
    vi.useFakeTimers();
    vi.mocked(authedMediaUrl)
      .mockRejectedValueOnce(new Error("temporary native transport failure"))
      .mockResolvedValueOnce("blob:recovered");
    const wrapper = mount(AuthedMedia, {
      props: {
        path: "/api/gallery/thumbnail/demo.png",
        target: { baseUrl: "http://local:7680", apiKey: "old-key" },
        cacheKey: "local",
      },
    });
    await flushPromises();
    await vi.advanceTimersByTimeAsync(250);

    expect(authedMediaUrl).toHaveBeenCalledTimes(2);
    expect(wrapper.get("img").attributes("src")).toBe("blob:recovered");
  });

  it("does not retry an unbounded full-media fetch", async () => {
    vi.useFakeTimers();
    vi.mocked(fullSizeMediaUrl).mockRejectedValue(new Error("video transfer failed"));
    const wrapper = mount(AuthedMedia, {
      props: { path: "/api/gallery/image/large-video.mp4", video: true },
    });
    await flushPromises();
    await vi.advanceTimersByTimeAsync(2_000);

    expect(fullSizeMediaUrl).toHaveBeenCalledTimes(1);
    expect(wrapper.text()).toContain("UNREADABLE");
  });

  it("reloads an unreadable thumbnail when its host route changes", async () => {
    vi.useFakeTimers();
    vi.mocked(authedMediaUrl).mockRejectedValue(new Error("host unavailable"));
    const wrapper = mount(AuthedMedia, {
      props: {
        path: "/api/gallery/thumbnail/demo.png",
        target: { baseUrl: "http://local:7680", apiKey: "old-key" },
        cacheKey: "local",
      },
    });
    await flushPromises();
    await vi.advanceTimersByTimeAsync(1_250);
    expect(wrapper.text()).toContain("UNREADABLE");

    vi.mocked(authedMediaUrl).mockResolvedValue("blob:reconnected");
    await wrapper.setProps({
      target: { baseUrl: "http://local:7680", apiKey: "new-key" },
    });
    await flushPromises();

    expect(wrapper.get("img").attributes("src")).toBe("blob:reconnected");
    expect(authedMediaUrl).toHaveBeenLastCalledWith(
      "/api/gallery/thumbnail/demo.png",
      expect.objectContaining({
        target: { baseUrl: "http://local:7680", apiKey: "new-key" },
      }),
    );
  });
});

describe("AuthedMedia target identity", () => {
  it("keeps the rendered element when a re-render passes an equal but new target object", async () => {
    // The Library builds `target` per render (`{ baseUrl, apiKey }`), so any
    // parent re-render — selecting a tile — hands every remote tile a fresh
    // object. Reloading on that swaps the <img> for the shimmer between the
    // two clicks of a double-click, which is why remote-only prints would not
    // open while This-Mac prints (a stable authority object) did.
    const wrapper = mount(AuthedMedia, {
      props: {
        path: "/api/gallery/thumbnail/remote.png",
        target: { baseUrl: "http://hal9000:7680", apiKey: null },
        cacheKey: "hal9000-7680",
      },
    });
    await vi.waitFor(() => expect(wrapper.find("img").exists()).toBe(true));
    const firstImg = wrapper.get("img").element;
    expect(authedMediaUrl).toHaveBeenCalledTimes(1);

    await wrapper.setProps({ target: { baseUrl: "http://hal9000:7680", apiKey: null } });
    await flushPromises();

    expect(authedMediaUrl).toHaveBeenCalledTimes(1);
    expect(wrapper.get("img").element).toBe(firstImg);
  });

  it("reloads when the target's URL or key actually changes", async () => {
    const wrapper = mount(AuthedMedia, {
      props: {
        path: "/api/gallery/thumbnail/remote.png",
        target: { baseUrl: "http://hal9000:7680", apiKey: null },
        cacheKey: "hal9000-7680",
      },
    });
    await vi.waitFor(() => expect(wrapper.find("img").exists()).toBe(true));

    await wrapper.setProps({ target: { baseUrl: "http://hal9000:7680", apiKey: "rotated" } });
    await flushPromises();
    expect(authedMediaUrl).toHaveBeenCalledTimes(2);

    await wrapper.setProps({ target: { baseUrl: "http://hal9000.local:7680", apiKey: "rotated" } });
    await flushPromises();
    expect(authedMediaUrl).toHaveBeenCalledTimes(3);
  });
});
