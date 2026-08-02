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

  it("recalls prompt history with ArrowUp/ArrowDown at the caret edges", async () => {
    const wrapper = factory({ prompt: "", history: ["newest", "older"] });
    const ta = wrapper.get("[data-test='composer-prompt']");
    const el = ta.element as HTMLTextAreaElement;
    el.selectionStart = 0;
    el.selectionEnd = 0;

    await ta.trigger("keydown", { key: "ArrowUp" });
    expect(wrapper.emitted("update:prompt")?.at(-1)?.[0]).toBe("newest");
    await ta.trigger("keydown", { key: "ArrowUp" });
    expect(wrapper.emitted("update:prompt")?.at(-1)?.[0]).toBe("older");
    await ta.trigger("keydown", { key: "ArrowDown" });
    expect(wrapper.emitted("update:prompt")?.at(-1)?.[0]).toBe("newest");
  });

  it("exposes record() to seed just-submitted prompts into recall", async () => {
    const wrapper = factory({ prompt: "", history: [] });
    (wrapper.vm as unknown as { record: (p: string) => void }).record(
      "fresh prompt",
    );
    const ta = wrapper.get("[data-test='composer-prompt']");
    const el = ta.element as HTMLTextAreaElement;
    el.selectionStart = 0;
    await ta.trigger("keydown", { key: "ArrowUp" });
    expect(wrapper.emitted("update:prompt")?.at(-1)?.[0]).toBe("fresh prompt");
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

  // Desktop parity (`ExpandControl`): expansion rewrites the prompt, so an
  // empty one has nothing to enrich — and that stays true on the surfaces
  // where a blank prompt is a legitimate render.
  it("disables Expand while the prompt is blank", async () => {
    const wrapper = factory({ prompt: "   " });
    const expand = wrapper.get("[data-test='composer-expand']");
    expect(expand.attributes("disabled")).toBeDefined();
    await wrapper.setProps({ prompt: "a lighthouse" });
    expect(expand.attributes("disabled")).toBeUndefined();
  });

  it("keeps Generate available while the prompt is blank", () => {
    const wrapper = factory({ prompt: "" });
    expect(
      wrapper.get("[data-test='composer-submit']").attributes("disabled"),
    ).toBeUndefined();
  });

  it("softens the placeholder once conditioning makes the prompt optional", async () => {
    const wrapper = factory();
    const prompt = wrapper.get("[data-test='composer-prompt']");
    expect(prompt.attributes("placeholder")).toBe(
      "Describe the image you want to create…",
    );
    await wrapper.setProps({ promptOptional: true });
    expect(prompt.attributes("placeholder")).toContain("optional");
  });
});
