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
      hasPrepared: false,
      chainReject: false,
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

  it("disables Generate without a prompt or with a prepared batch", () => {
    const emptyForm = baseForm();
    emptyForm.prompt = "";
    expect(
      mountComposer(emptyForm).get("[data-test='generate-button']").attributes("disabled"),
    ).toBeDefined();
    expect(
      mountComposer(baseForm(), { hasPrepared: true })
        .get("[data-test='generate-button']")
        .attributes("disabled"),
    ).toBeDefined();
    expect(
      mountComposer(baseForm(), { submitting: true })
        .get("[data-test='generate-button']")
        .attributes("disabled"),
    ).toBeDefined();
  });

  it("hides the Generate button for prepared multi-image batches", () => {
    const wrapper = mountComposer(baseForm(), { effectiveBatchSize: 3 });
    expect(wrapper.find("[data-test='generate-button']").exists()).toBe(false);
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
