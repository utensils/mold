import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import AuthedMedia from "./AuthedMedia.vue";
import { authedMediaUrl } from "../../lib/gallery/media";

vi.mock("../../lib/gallery/media", () => ({
  authedMediaUrl: vi.fn().mockResolvedValue("blob:video"),
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(authedMediaUrl).mockResolvedValue("blob:video");
});

afterEach(() => vi.useRealTimers());

describe("AuthedMedia", () => {
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
    vi.mocked(authedMediaUrl).mockRejectedValue(new Error("video transfer failed"));
    const wrapper = mount(AuthedMedia, {
      props: { path: "/api/gallery/image/large-video.mp4", video: true },
    });
    await flushPromises();
    await vi.advanceTimersByTimeAsync(2_000);

    expect(authedMediaUrl).toHaveBeenCalledTimes(1);
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
