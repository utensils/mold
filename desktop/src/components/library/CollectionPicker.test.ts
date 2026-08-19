import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import CollectionPicker from "./CollectionPicker.vue";
import type { MergedCollection } from "@studio/lib/libraryOrganization";

const merged = (slug: string, name: string, count = 0): MergedCollection => ({
  slug,
  name,
  count,
  hosts: [{ hostId: "local", id: `id-${slug}`, count }],
  cover: null,
});

const collections = [
  merged("smurfs", "Smurfs", 9),
  merged("river-studies", "River studies", 6),
  merged("client-halcyon", "Client · Halcyon", 5),
];

function mountPicker(extra: Record<string, unknown> = {}) {
  return mount(CollectionPicker, {
    props: { collections, selected: ["smurfs"], mixed: ["river-studies"], ...extra },
    attachTo: document.body,
  });
}

describe("CollectionPicker", () => {
  it("renders one checkbox row per collection with checked / mixed / unchecked states", () => {
    const wrapper = mountPicker();
    const rows = wrapper.findAll("[data-test='collection-row']");
    expect(rows.map((r) => r.attributes("aria-checked"))).toEqual(["true", "mixed", "false"]);
    expect(rows.map((r) => r.attributes("role"))).toEqual(["checkbox", "checkbox", "checkbox"]);
    expect(rows[0]!.find("[data-test='collection-box']").text()).toBe("✓");
    expect(rows[1]!.find("[data-test='collection-box']").text()).toBe("–");
    expect(rows[0]!.text()).toContain("9");
    expect(wrapper.find("[role='group']").attributes("aria-label")).toBe("Collections");
    wrapper.unmount();
  });

  it("toggle emits (slug, checked): unchecked → true, mixed → true, checked → false", async () => {
    const wrapper = mountPicker();
    const rows = wrapper.findAll("[data-test='collection-row']");
    await rows[2]!.trigger("click");
    await rows[1]!.trigger("click");
    await rows[0]!.trigger("click");
    expect(wrapper.emitted("toggle")).toEqual([
      ["client-halcyon", true],
      ["river-studies", true],
      ["smurfs", false],
    ]);
    wrapper.unmount();
  });

  it("uses the counts override when given", () => {
    const wrapper = mountPicker({ counts: (slug: string) => (slug === "smurfs" ? 2 : 0) });
    const rows = wrapper.findAll("[data-test='collection-row']");
    expect(rows[0]!.text()).toContain("2");
    expect(rows[1]!.text()).not.toContain("6");
    wrapper.unmount();
  });

  it("New collection… turns into an inline input; Enter creates, empty / Escape cancels", async () => {
    const wrapper = mountPicker();
    await wrapper.find("[data-test='collection-new']").trigger("click");
    const input = wrapper.find<HTMLInputElement>("[data-test='collection-new-input']");
    expect(input.exists()).toBe(true);
    await input.setValue("  Film grain tests ");
    await input.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("create")).toEqual([["Film grain tests"]]);
    expect(wrapper.find("[data-test='collection-new-input']").exists()).toBe(false);

    await wrapper.find("[data-test='collection-new']").trigger("click");
    await wrapper.find("[data-test='collection-new-input']").setValue("   ");
    await wrapper.find("[data-test='collection-new-input']").trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("create")).toHaveLength(1);

    await wrapper.find("[data-test='collection-new']").trigger("click");
    await wrapper.find("[data-test='collection-new-input']").setValue("x");
    await wrapper.find("[data-test='collection-new-input']").trigger("keydown", { key: "Escape" });
    expect(wrapper.emitted("create")).toHaveLength(1);
    expect(wrapper.find("[data-test='collection-new-input']").exists()).toBe(false);
    wrapper.unmount();
  });

  it("hides the create row when allowCreate is false and shows an empty hint with no collections", () => {
    const wrapper = mount(CollectionPicker, {
      props: { collections: [], selected: [], allowCreate: false },
    });
    expect(wrapper.find("[data-test='collection-new']").exists()).toBe(false);
    expect(wrapper.find("[data-test='collection-picker-empty']").text()).toBe(
      "No collections yet.",
    );
    wrapper.unmount();
  });

  it("names the fan-out hosts and disables every row when disabled", async () => {
    const wrapper = mountPicker({ hostNote: "fans out to This Mac · plato", disabled: true });
    expect(wrapper.find("[data-test='collection-host-note']").text()).toBe(
      "fans out to This Mac · plato",
    );
    await wrapper.find("[data-test='collection-row']").trigger("click");
    expect(wrapper.emitted("toggle")).toBeUndefined();
    expect(wrapper.find("[data-test='collection-new']").attributes("disabled")).toBeDefined();
    wrapper.unmount();
  });
});
