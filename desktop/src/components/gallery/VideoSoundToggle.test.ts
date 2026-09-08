import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises, mount } from "@vue/test-utils";
import { installMemoryLocalStorage } from "../../lib/testSupport/memoryLocalStorage";
import { useVideoPlaybackStore, VIDEO_MUTED_STORAGE_KEY } from "../../stores/videoPlayback";
import VideoSoundToggle from "./VideoSoundToggle.vue";
import AuthedMedia from "./AuthedMedia.vue";

vi.mock("../../lib/gallery/media", async (original) => ({
  ...(await original<typeof import("../../lib/gallery/media")>()),
  fullSizeMediaUrl: vi.fn().mockResolvedValue("blob:clip"),
}));

installMemoryLocalStorage();
beforeEach(() => {
  localStorage.removeItem(VIDEO_MUTED_STORAGE_KEY);
  setActivePinia(createPinia());
});

describe("video sound preference", () => {
  it("keeps both controls and a looping player in sync, and remembers mute for the next launch", async () => {
    const first = mount(VideoSoundToggle);
    const second = mount(VideoSoundToggle);
    const player = mount(AuthedMedia, {
      props: { path: "/api/gallery/image/clip.mp4", video: true },
    });
    await flushPromises();
    const video = player.get("video").element as HTMLVideoElement;
    expect(first.text()).toBe("Sound on");
    await first.trigger("click");
    expect(second.text()).toBe("Sound off");
    expect(second.attributes("aria-pressed")).toBe("true");
    expect(video.muted).toBe(true);
    expect(video.loop).toBe(true);
    expect(localStorage.getItem(VIDEO_MUTED_STORAGE_KEY)).toBe("true");
    setActivePinia(createPinia());
    expect(useVideoPlaybackStore().muted).toBe(true);
    first.unmount();
    second.unmount();
    player.unmount();
  });

  it("reflects native player sound changes and can unmute a zero-volume player", async () => {
    const toggle = mount(VideoSoundToggle);
    const player = mount(AuthedMedia, {
      props: { path: "/api/gallery/image/clip.mp4", video: true },
    });
    await flushPromises();
    const video = player.get("video").element as HTMLVideoElement;
    video.volume = 0;
    await player.get("video").trigger("volumechange");
    expect(toggle.text()).toBe("Sound off");
    await toggle.trigger("click");
    expect(video.volume).toBe(1);
    expect(video.muted).toBe(false);
    video.muted = true;
    await player.get("video").trigger("volumechange");
    expect(toggle.text()).toBe("Sound off");
    video.muted = false;
    await player.get("video").trigger("volumechange");
    expect(toggle.text()).toBe("Sound on");
    toggle.unmount();
    player.unmount();
  });
});
