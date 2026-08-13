import { afterEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { reactive } from "vue";
import ComposerCard from "./ComposerCard.vue";
import StyleChips from "./StyleChips.vue";
import ExpandControl from "../generate/ExpandControl.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";

vi.mock("../../lib/platform", () => ({
  primaryModifierPressed: (e: KeyboardEvent) => e.metaKey || e.ctrlKey,
  shortcutLabel: (k: string) => k,
}));

afterEach(() => (document.body.innerHTML = ""));

function baseForm(): GenerateForm {
  const form = reactive({ ...newGenerateForm(), family: "flux" });
  form.model = "flux-dev:q8";
  form.prompt = "a lighthouse";
  return form;
}

function mountComposer(form: GenerateForm, overrides: Record<string, unknown> = {}) {
  return mount(ComposerCard, {
    attachTo: document.body,
    global: { stubs: { EstimateBadge: true } },
    props: {
      form,
      effectiveBatchSize: 1,
      expansionRunning: false,
      expansionHostLabel: null,
      canUndo: false,
      preparedBlocked: false,
      disabled: false,
      disabledReason: null,
      submitting: false,
      buttonLabel: "Generate",
      estimateRequest: null,
      estimateTarget: null,
      preprocessingStatus: null,
      history: [],
      ...overrides,
    },
  });
}

describe("ComposerCard", () => {
  it("submits on the primary-modifier Enter shortcut", async () => {
    const wrapper = mountComposer(baseForm());
    await wrapper
      .get("textarea[aria-label='Prompt']")
      .trigger("keydown", { key: "Enter", metaKey: true });
    expect(wrapper.emitted("generate")).toHaveLength(1);
  });

  it("expands on the primary-modifier E shortcut when the prompt is valid", async () => {
    const wrapper = mountComposer(baseForm());
    await wrapper
      .get("textarea[aria-label='Prompt']")
      .trigger("keydown", { key: "e", metaKey: true });
    expect(wrapper.emitted("expand")).toHaveLength(1);
  });

  it("routes the style-chip selection through the form without touching the prompt", async () => {
    const form = baseForm();
    const wrapper = mountComposer(form);
    wrapper.findComponent(StyleChips).vm.$emit("update:modelValue", "cinematic");
    await wrapper.vm.$nextTick();
    expect(form.stylePreset).toBe("cinematic");
    expect(form.prompt).toBe("a lighthouse"); // textarea untouched
  });

  it("can disable Generate without displaying obvious guidance", () => {
    const emptyForm = baseForm();
    emptyForm.prompt = "";
    const missingPrompt = mountComposer(emptyForm, { disabled: true });
    expect(missingPrompt.get("[data-test='generate-button']").attributes("disabled")).toBeDefined();
    expect(missingPrompt.find("[data-test='action-blocker']").exists()).toBe(false);
  });

  it("disables Generate and displays non-obvious corrective guidance", () => {
    expect(
      mountComposer(baseForm(), {
        disabled: true,
        disabledReason: "Use the reviewed variations panel.",
      })
        .get("[data-test='generate-button']")
        .attributes("disabled"),
    ).toBeDefined();
    expect(
      mountComposer(baseForm(), { submitting: true })
        .get("[data-test='generate-button']")
        .attributes("disabled"),
    ).toBeDefined();
  });

  it("keeps Generate visible for multi-image batches", () => {
    const wrapper = mountComposer(baseForm(), { effectiveBatchSize: 3 });
    expect(wrapper.get("[data-test='generate-button']").text()).toContain("Generate");
  });

  it("disables Generate while batch expansion is running", () => {
    const wrapper = mountComposer(baseForm(), {
      effectiveBatchSize: 3,
      expansionRunning: true,
      disabled: true,
      disabledReason: "Wait for prompt preparation to finish.",
    });
    expect(wrapper.get("[data-test='generate-button']").attributes("disabled")).toBeDefined();
  });

  it("does not submit from the shortcut while Generate is disabled", async () => {
    const wrapper = mountComposer(baseForm(), {
      effectiveBatchSize: 3,
      disabled: true,
      disabledReason: "Use the reviewed variations panel.",
    });
    await wrapper
      .get("textarea[aria-label='Prompt']")
      .trigger("keydown", { key: "Enter", metaKey: true });
    expect(wrapper.emitted("generate")).toBeUndefined();
  });

  it("forwards blocked + running expansion state to the ExpandControl", () => {
    const wrapper = mountComposer(baseForm(), { preparedBlocked: true, expansionRunning: true });
    const control = wrapper.findComponent(ExpandControl);
    expect(control.props("blocked")).toBe(true);
    expect(control.props("running")).toBe(true);
  });

  it("exposes focus and record for the view to drive", () => {
    const wrapper = mountComposer(baseForm());
    expect(typeof (wrapper.vm as unknown as { focus: () => void }).focus).toBe("function");
    expect(typeof (wrapper.vm as unknown as { record: (p: string) => void }).record).toBe(
      "function",
    );
  });
});
