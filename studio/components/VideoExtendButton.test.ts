import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, expect, it, vi } from "vitest";
import { VIDEO_LIBRARY_EXTENSION_NAVIGATION } from "../lib/videoLibraryExtensionHandoff";
const api = vi.hoisted(() => ({ caps: vi.fn(), inspect: vi.fn() }));
vi.mock("../api/client", () => ({ apiJsonTo: api.caps }));
vi.mock("../api/videoLibraryExtension", () => ({
  inspectVideoLibraryExtension: api.inspect,
}));
import VideoExtendButton from "./VideoExtendButton.vue";
const owner = {
  hostId: "plato",
  label: "Plato",
  instanceId: "install",
  target: { baseUrl: "http://plato", apiKey: "private" },
};
beforeEach(() => {
  vi.clearAllMocks();
  api.caps.mockResolvedValue({
    video_jobs: {
      supported: true,
      literal_prompts: true,
      library_extend: true,
    },
  });
});
it("offers ordinary video Extend only on an affirmative host", async () => {
  api.caps.mockResolvedValue({});
  const wrapper = mount(VideoExtendButton, {
    props: { owner, filename: "ordinary.mp4", metadata: {} },
    global: {
      provide: { [VIDEO_LIBRARY_EXTENSION_NAVIGATION as symbol]: vi.fn() },
    },
  });
  await flushPromises();
  expect(wrapper.find("button").exists()).toBe(false);
  expect(api.inspect).not.toHaveBeenCalled();
  wrapper.unmount();
});
it("inspects the captured ordinary Library video before navigating without downloading a prefix", async () => {
  const navigate = vi.fn(),
    selection = {
      owner,
      source: { filename: "ordinary.mp4" },
      title: "Ordinary",
      metadata: { prompt: "first" },
      inventory: { availability: "available", members: [] },
    };
  api.inspect.mockResolvedValue(selection);
  const wrapper = mount(VideoExtendButton, {
    props: { owner, filename: "ordinary.mp4", metadata: { prompt: "first" } },
    global: {
      provide: { [VIDEO_LIBRARY_EXTENSION_NAVIGATION as symbol]: navigate },
    },
  });
  await flushPromises();
  await wrapper.get("button").trigger("click");
  await flushPromises();
  expect(api.inspect.mock.calls[0]?.[1]).toBe("ordinary.mp4");
  expect(navigate).toHaveBeenCalledWith(selection);
  wrapper.unmount();
});
it("keeps a pending Extend inspection alive across an equivalent polled owner object", async () => {
  let finish!: (value: unknown) => void;
  api.inspect.mockReturnValue(
    new Promise((resolve) => {
      finish = resolve;
    }),
  );
  const navigate = vi.fn();
  const wrapper = mount(VideoExtendButton, {
    props: { owner, filename: "ordinary.mp4", metadata: {} },
    global: {
      provide: { [VIDEO_LIBRARY_EXTENSION_NAVIGATION as symbol]: navigate },
    },
  });
  await flushPromises();
  await wrapper.get("button").trigger("click");
  await wrapper.setProps({ owner: { ...owner, target: { ...owner.target } } });
  await flushPromises();
  const selected = {
    owner,
    source: { filename: "ordinary.mp4" },
    metadata: {},
    title: "Original",
    inventory: { availability: "available", members: [] },
  };
  finish(selected);
  await flushPromises();
  expect(api.caps).toHaveBeenCalledTimes(1);
  expect(navigate).toHaveBeenCalledWith(selected);
  wrapper.unmount();
});
