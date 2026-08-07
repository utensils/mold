import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

const { mediaSaveDirectory, pickDirectory, appSettingsGet, appSettingsSet } = vi.hoisted(() => ({
  mediaSaveDirectory: vi.fn(),
  pickDirectory: vi.fn(),
  appSettingsGet: vi.fn(),
  appSettingsSet: vi.fn(),
}));

vi.mock("../../lib/ipc", () => ({
  ipc: { mediaSaveDirectory, pickDirectory, appSettingsGet, appSettingsSet },
}));

import MediaSection from "./MediaSection.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useToastStore } from "../../stores/toasts";

describe("MediaSection", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    mediaSaveDirectory.mockReset().mockResolvedValue("/Users/test/Downloads");
    pickDirectory.mockReset();
    appSettingsSet.mockReset().mockResolvedValue(undefined);
    appSettingsGet.mockReset().mockResolvedValue({ mediaSaveDir: null });
    useAppPrefsStore().settings = { mediaSaveDir: null } as never;
  });

  it("shows the effective system default and explains which actions use it", async () => {
    const wrapper = mount(MediaSection);
    await flushPromises();

    expect(wrapper.get("[data-test='media-save-directory']").text()).toBe("/Users/test/Downloads");
    expect(wrapper.text()).toContain("Save image, Save video, and converted exports go here");
    expect(wrapper.text()).toContain("System default");
  });

  it("persists a chosen folder and confirms the change", async () => {
    pickDirectory.mockResolvedValue("/Volumes/Studio/Exports");
    mediaSaveDirectory
      .mockResolvedValueOnce("/Users/test/Downloads")
      .mockResolvedValueOnce("/Volumes/Studio/Exports");
    const wrapper = mount(MediaSection);
    await flushPromises();

    await wrapper.get("[data-test='choose-media-save-directory']").trigger("click");
    await flushPromises();

    expect(appSettingsSet).toHaveBeenCalledWith(
      expect.objectContaining({ mediaSaveDir: "/Volumes/Studio/Exports" }),
    );
    expect(wrapper.get("[data-test='media-save-directory']").text()).toBe(
      "/Volumes/Studio/Exports",
    );
    expect(useToastStore().items.at(-1)?.message).toBe("Save location updated");
  });
});
