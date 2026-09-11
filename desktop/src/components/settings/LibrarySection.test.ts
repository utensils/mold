import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import LibrarySection from "./LibrarySection.vue";
import ToggleControl from "@studio/components/settings/ToggleControl.vue";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useLibraryPrefsStore } from "../../stores/libraryPrefs";
import { useGenerateFormStore } from "../../stores/generateForm";
import { AUTO_TAG_TITLE_STORAGE_KEY } from "../../lib/libraryPrefs";
import { installMemoryLocalStorage } from "../../lib/testSupport/memoryLocalStorage";

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { pickDirectory: vi.fn(() => Promise.resolve(null)) },
}));
vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
  ApiError: class extends Error {},
}));

installMemoryLocalStorage();

beforeEach(() => {
  setActivePinia(createPinia());
  localStorage.clear();
});

describe("Settings ▸ Library", () => {
  it("offers the title auto-tag toggle, on by default", () => {
    useLibraryPrefsStore().init();
    const wrapper = mount(LibrarySection);
    const toggle = wrapper.getComponent(ToggleControl);
    expect(wrapper.text()).toContain("Tag new prints with their title");
    expect(toggle.props("modelValue")).toBe(true);
  });

  it("persists the change and re-mirrors it onto the Create form", async () => {
    const prefs = useLibraryPrefsStore();
    const form = useGenerateFormStore();
    prefs.init();
    const wrapper = mount(LibrarySection);
    await wrapper.getComponent(ToggleControl).vm.$emit("commit", false);
    expect(prefs.autoTagTitle).toBe(false);
    expect(localStorage.getItem(AUTO_TAG_TITLE_STORAGE_KEY)).toBe("false");
    expect(form.form.fileUnderAutoTag).toBe(false);
  });

  it("still edits this device's trash retention", () => {
    useLibraryPrefsStore().init();
    const wrapper = mount(LibrarySection);
    expect(wrapper.find("[data-test='library-remote-note']").exists()).toBe(true);
  });

  /*
   * Two switches read almost identically and govern different things: the
   * engine key is what `mold run` puts in a request, the app one is what this
   * Create form offers. They sit adjacent, app first, so the difference is
   * read as a pair rather than found twice in different parts of the page.
   */
  it("puts the engine's command-line auto-tag key directly beneath the app's own", () => {
    useLibraryPrefsStore().init();
    useSettingsConfigStore().rows = [
      {
        key: "generate.auto_tag_title",
        value: false,
        source: "db",
        env_var: null,
        restart_required: false,
      },
    ];
    const wrapper = mount(LibrarySection);
    const labels = wrapper.findAll(".ms-setting-row__label").map((el) => el.text());
    const app = labels.indexOf("Tag new prints with their title");
    expect(app).toBeGreaterThanOrEqual(0);
    expect(labels[app + 1]).toBe("Tag command-line prints with their title");
    expect(wrapper.text()).toContain("Each app keeps its own switch");
  });
});
