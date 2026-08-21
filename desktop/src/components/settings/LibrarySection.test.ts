import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import LibrarySection from "./LibrarySection.vue";
import ToggleControl from "./ToggleControl.vue";
import { useLibraryPrefsStore } from "../../stores/libraryPrefs";
import { useGenerateFormStore } from "../../stores/generateForm";
import { AUTO_TAG_TITLE_STORAGE_KEY } from "../../lib/libraryPrefs";
import { installMemoryLocalStorage } from "../../lib/testSupport/memoryLocalStorage";

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
});
