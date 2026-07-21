import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ComposerCard from "./ComposerCard.vue";

function factory(
  props: Partial<InstanceType<typeof ComposerCard>["$props"]> = {},
) {
  return mount(ComposerCard, {
    props: {
      prompt: "a lighthouse",
      stylePreset: null,
      aspectLabel: "1:1",
      width: 1024,
      height: 1024,
      steps: 28,
      batchSize: 1,
      ...props,
    },
  });
}

describe("ComposerCard", () => {
  it("submits on ⌘↵ / Ctrl+↵ inside the textarea", async () => {
    const wrapper = factory();
    const ta = wrapper.get("[data-test='composer-prompt']");
    await ta.trigger("keydown", { key: "Enter", metaKey: true });
    expect(wrapper.emitted("submit")).toHaveLength(1);

    await ta.trigger("keydown", { key: "Enter", ctrlKey: true });
    expect(wrapper.emitted("submit")).toHaveLength(2);
  });

  it("does not submit on a plain Enter", async () => {
    const wrapper = factory();
    await wrapper
      .get("[data-test='composer-prompt']")
      .trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("submit")).toBeUndefined();
  });

  it("does not submit while busy", async () => {
    const wrapper = factory({ busy: true });
    await wrapper
      .get("[data-test='composer-prompt']")
      .trigger("keydown", { key: "Enter", metaKey: true });
    expect(wrapper.emitted("submit")).toBeUndefined();
  });

  it("emits the prompt on input without touching style", async () => {
    const wrapper = factory();
    const ta = wrapper.get("[data-test='composer-prompt']")
      .element as HTMLTextAreaElement;
    ta.value = "a lighthouse in a storm";
    await wrapper.get("[data-test='composer-prompt']").trigger("input");
    expect(wrapper.emitted("update:prompt")?.[0]).toEqual([
      "a lighthouse in a storm",
    ]);
  });

  it("selects a style preset, and deselects it when tapped while active", async () => {
    const wrapper = factory();
    await wrapper.get("[data-test='style-toggle']").trigger("click");
    await wrapper.get("[data-test='style-chip-cinematic']").trigger("click");
    expect(wrapper.emitted("update:stylePreset")?.[0]).toEqual(["cinematic"]);

    await wrapper.setProps({ stylePreset: "cinematic" });
    await wrapper.get("[data-test='style-chip-cinematic']").trigger("click");
    expect(wrapper.emitted("update:stylePreset")?.[1]).toEqual([null]);
  });

  it("shows the active preset name in the collapsed style chip", async () => {
    const wrapper = factory({ stylePreset: "anime" });
    expect(wrapper.get("[data-test='style-active']").text()).toBe("Anime");
  });

  it("renders the summary line, adding ×N only for a batch", () => {
    expect(factory().get("[data-test='composer-summary']").text()).toBe(
      "1:1 · 1024×1024 · 28 steps",
    );
    const batched = factory({ batchSize: 3 });
    expect(batched.get("[data-test='composer-summary']").text()).toBe(
      "1:1 · 1024×1024 · 28 steps · ×3",
    );
  });

  it("labels Expand for the current batch size", () => {
    expect(factory().get("[data-test='composer-expand']").text()).toContain(
      "Expand prompt",
    );
    expect(
      factory({ batchSize: 4 }).get("[data-test='composer-expand']").text(),
    ).toContain("Expand to 4");
  });

  it("shows the undo affordance only when an expansion is undoable", async () => {
    const wrapper = factory();
    expect(wrapper.find("[data-test='composer-undo']").exists()).toBe(false);
    await wrapper.setProps({ expanded: true });
    expect(wrapper.find("[data-test='composer-undo']").exists()).toBe(true);
    await wrapper.get("[data-test='composer-undo']").trigger("click");
    expect(wrapper.emitted("undo-expand")).toHaveLength(1);
  });
});
