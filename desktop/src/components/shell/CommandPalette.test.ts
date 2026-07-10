import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../../lib/api/history", () => ({ fetchHistory: vi.fn().mockResolvedValue([]) }));
vi.mock("../../lib/api/models", () => ({ loadModel: vi.fn(), unloadModel: vi.fn() }));

import CommandPalette from "./CommandPalette.vue";
import { useUiStore } from "../../stores/ui";

beforeEach(() => {
  setActivePinia(createPinia());
});

async function openPalette() {
  const wrapper = mount(CommandPalette, { attachTo: document.body });
  const ui = useUiStore();
  ui.paletteOpen = true;
  await wrapper.vm.$nextTick();
  await wrapper.vm.$nextTick();
  return wrapper;
}

describe("CommandPalette a11y semantics", () => {
  it("is a modal dialog wrapping a combobox and listbox", async () => {
    const wrapper = await openPalette();

    const dialog = wrapper.get("[role='dialog']");
    expect(dialog.attributes("aria-modal")).toBe("true");
    expect(dialog.attributes("aria-label")).toBe("Command palette");

    const combobox = wrapper.get("[role='combobox']");
    expect(combobox.attributes("aria-controls")).toBe("cmd-palette-listbox");
    expect(combobox.attributes("aria-expanded")).toBe("true");

    const listbox = wrapper.get("#cmd-palette-listbox");
    expect(listbox.attributes("role")).toBe("listbox");
    wrapper.unmount();
  });

  it("marks options with role=option and points aria-activedescendant at the selection", async () => {
    const wrapper = await openPalette();

    const options = wrapper.findAll("[role='option']");
    expect(options.length).toBeGreaterThan(0);
    expect(options[0]!.attributes("aria-selected")).toBe("true");
    expect(options[0]!.attributes("id")).toBe("cmd-palette-option-0");

    // The combobox's active descendant tracks the highlighted option.
    expect(wrapper.get("[role='combobox']").attributes("aria-activedescendant")).toBe(
      "cmd-palette-option-0",
    );
    wrapper.unmount();
  });
});
