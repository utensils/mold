import { describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import TagEditor from "./TagEditor.vue";

const suggestions = [
  { name: "blue", count: 9 },
  { name: "Blueprint", count: 2 },
  { name: "portrait", count: 5 },
  { name: "keep", count: 1 },
];

function mountEditor(modelValue: string[] = ["blue"], extra: Record<string, unknown> = {}) {
  return mount(TagEditor, {
    props: { modelValue, suggestions, ...extra },
    attachTo: document.body,
  });
}

const input = (wrapper: ReturnType<typeof mountEditor>) =>
  wrapper.find<HTMLInputElement>("[data-test='tag-input']");

const key = async (wrapper: ReturnType<typeof mountEditor>, k: string) => {
  await input(wrapper).trigger("keydown", { key: k });
};

describe("TagEditor", () => {
  it("renders one chip per tag with a remove button, and a combobox input on a ≥34px row", () => {
    const wrapper = mountEditor(["blue", "portrait"]);
    expect(
      wrapper.findAll("[data-test='tag-chip']").map((c) => c.text().replace(/\s*×$/, "")),
    ).toEqual(["blue", "portrait"]);
    expect(wrapper.find("[data-test='tag-remove']").attributes("aria-label")).toBe(
      "Remove tag blue",
    );
    const box = input(wrapper);
    expect(box.attributes("role")).toBe("combobox");
    expect(box.attributes("aria-autocomplete")).toBe("list");
    expect(wrapper.find("[data-test='tag-editor']").classes()).toContain("min-h-[34px]");
    wrapper.unmount();
  });

  it("Enter adds the normalized draft and emits update + add", async () => {
    const wrapper = mountEditor(["blue"]);
    await input(wrapper).setValue("  #Portrait ");
    await key(wrapper, "Enter");
    // A literal leading hash addresses a distinct tag, matching the server.
    expect(wrapper.emitted("update:modelValue")).toEqual([[["blue", "#Portrait"]]]);
    expect(wrapper.emitted("add")).toEqual([["#Portrait"]]);
    expect(input(wrapper).element.value).toBe("");
    wrapper.unmount();
  });

  it("comma adds too, and a duplicate (case-insensitive) is dropped silently", async () => {
    const wrapper = mountEditor(["blue"]);
    await input(wrapper).setValue("keep");
    await key(wrapper, ",");
    expect(wrapper.emitted("add")).toEqual([["keep"]]);
    await input(wrapper).setValue("BLUE");
    await key(wrapper, "Enter");
    expect(wrapper.emitted("add")).toHaveLength(1);
    expect(input(wrapper).element.value).toBe("");
    wrapper.unmount();
  });

  it("Backspace on an empty input removes the last chip; × removes a specific one", async () => {
    const wrapper = mountEditor(["blue", "portrait"]);
    await key(wrapper, "Backspace");
    expect(wrapper.emitted("remove")).toEqual([["portrait"]]);
    expect(wrapper.emitted("update:modelValue")).toEqual([[["blue"]]]);
    await wrapper.find("[data-test='tag-remove']").trigger("click");
    expect(wrapper.emitted("remove")).toEqual([["portrait"], ["blue"]]);
    wrapper.unmount();
  });

  it("suggests by case-insensitive key, excludes present tags, ↓ highlights and Enter picks", async () => {
    const wrapper = mountEditor(["blue"]);
    await input(wrapper).setValue("blu");
    const options = wrapper.findAll("[role='option']");
    // "blue" is already on the print; only Blueprint matches.
    expect(options.map((o) => o.text())).toEqual(["Blueprint2"]);
    expect(input(wrapper).attributes("aria-expanded")).toBe("true");
    await key(wrapper, "ArrowDown");
    expect(wrapper.find("[role='option']").attributes("aria-selected")).toBe("true");
    expect(input(wrapper).attributes("aria-activedescendant")).toBeDefined();
    await key(wrapper, "Enter");
    expect(wrapper.emitted("add")).toEqual([["Blueprint"]]);
    wrapper.unmount();
  });

  it("Escape closes the suggestions without leaving the editor", async () => {
    const wrapper = mountEditor([]);
    await input(wrapper).trigger("focus");
    await input(wrapper).setValue("p");
    expect(wrapper.find("[data-test='tag-suggestions']").exists()).toBe(true);
    await key(wrapper, "Escape");
    expect(wrapper.find("[data-test='tag-suggestions']").exists()).toBe(false);
    expect(input(wrapper).element.value).toBe("p");
    wrapper.unmount();
  });

  it("clicking a suggestion adds it", async () => {
    const wrapper = mountEditor([]);
    await input(wrapper).trigger("focus");
    await input(wrapper).setValue("port");
    await wrapper.find("[role='option']").trigger("mousedown");
    expect(wrapper.emitted("add")).toEqual([["portrait"]]);
    wrapper.unmount();
  });

  it("disabled: no edits", async () => {
    const onAdd = vi.fn();
    const wrapper = mountEditor(["blue"], { disabled: true, onAdd });
    expect(input(wrapper).attributes("disabled")).toBeDefined();
    await key(wrapper, "Backspace");
    expect(wrapper.emitted("remove")).toBeUndefined();
    expect(wrapper.find("[data-test='tag-remove']").attributes("disabled")).toBeDefined();
    wrapper.unmount();
  });
});
