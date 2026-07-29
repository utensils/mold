import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import AuthedMedia from "./AuthedMedia.vue";

vi.mock("../../lib/gallery/media", () => ({
  authedMediaUrl: vi.fn().mockResolvedValue("blob:video"),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

describe("AuthedMedia", () => {
  it("disables Picture-in-Picture for desktop video playback", async () => {
    const wrapper = mount(AuthedMedia, {
      props: { path: "/api/gallery/video/demo.mp4", video: true, controls: true },
    });
    await vi.waitFor(() => expect(wrapper.find("video").exists()).toBe(true));

    expect(wrapper.get("video").attributes()).toHaveProperty("disablepictureinpicture");
  });
});
