import { afterEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { reactive } from "vue";
import ComposerCard from "./ComposerCard.vue";
import StyleChips from "./StyleChips.vue";
import ExpandControl from "../generate/ExpandControl.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { recipeCapabilitiesSnapshot } from "../../lib/capabilities";
import { hunyuan3dRecipe, sdxlRecipe } from "@studio/lib/generationProfile.testFixtures";
import { IGNORED_PROMPT_PLACEHOLDER } from "@studio/lib/promptRequirement";
import { PROMPT_IGNORED_TRANSFORM_REASON } from "@studio/lib/promptTransform";

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
  it("walks prompt history chronologically and restores the authored draft", async () => {
    const form = baseForm();
    form.prompt = "draft";
    const wrapper = mountComposer(form, { history: ["newest", "middle", "oldest"] });
    const textarea = wrapper.get("textarea[aria-label='Prompt']");
    const el = textarea.element as HTMLTextAreaElement;
    el.setSelectionRange(0, 0);

    await textarea.trigger("keydown", { key: "ArrowUp" });
    expect(form.prompt).toBe("newest");
    await textarea.trigger("keydown", { key: "ArrowUp" });
    expect(form.prompt).toBe("middle");
    await textarea.trigger("keydown", { key: "ArrowDown" });
    expect(form.prompt).toBe("newest");
    await textarea.trigger("keydown", { key: "ArrowDown" });
    expect(form.prompt).toBe("draft");
  });

  it("tags a ↑/↓ history recall so the view can release a quick expansion", async () => {
    const form = baseForm();
    form.prompt = "storm light";
    const wrapper = mountComposer(form, { history: ["newest", "oldest"] });
    const textarea = wrapper.get("textarea[aria-label='Prompt']");
    const el = textarea.element as HTMLTextAreaElement;
    el.setSelectionRange(0, 0);

    await textarea.trigger("keydown", { key: "ArrowUp" });
    expect(wrapper.emitted("prompt-authored")?.at(-1)).toEqual(["newest", "recalled"]);
    await textarea.trigger("keydown", { key: "ArrowDown" });
    // Walking back to the draft is still a recall: the draft text returns as
    // the user's own prompt, not as the prepared rewrite it used to be.
    expect(wrapper.emitted("prompt-authored")?.at(-1)).toEqual(["storm light", "recalled"]);
  });

  it("tags hand edits as typing so a quick expansion keeps its stale recovery", async () => {
    const wrapper = mountComposer(baseForm());
    await wrapper.get("textarea[aria-label='Prompt']").setValue("a lighthouse, edited");
    expect(wrapper.emitted("prompt-authored")?.at(-1)).toEqual(["a lighthouse, edited", "typed"]);
  });

  it("does not swallow ArrowUp when there is no cached or live history", () => {
    const wrapper = mountComposer(baseForm(), { history: [] });
    const el = wrapper.get("textarea[aria-label='Prompt']").element as HTMLTextAreaElement;
    el.setSelectionRange(0, 0);
    const event = new KeyboardEvent("keydown", { key: "ArrowUp", cancelable: true });
    el.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });

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

  it("disables blocked Generate but keeps an in-flight submission cancellable", async () => {
    expect(
      mountComposer(baseForm(), {
        disabled: true,
        disabledReason: "Use the reviewed variations panel.",
      })
        .get("[data-test='generate-button']")
        .attributes("disabled"),
    ).toBeDefined();
    const submitting = mountComposer(baseForm(), { submitting: true });
    const cancel = submitting.get("[data-test='generate-button']");
    expect(cancel.attributes("disabled")).toBeUndefined();
    await cancel.trigger("click");
    expect(submitting.emitted("cancel")).toHaveLength(1);
  });

  it("shows a non-blocking size advisory while Generate stays enabled", () => {
    const wrapper = mountComposer(baseForm(), {
      warningReason: "This model expects at least 1344 × 768 — the server may reject this size.",
    });
    const blocker = wrapper.get("[data-test='action-blocker']");
    expect(blocker.attributes("data-variant")).toBe("warn");
    expect(blocker.text()).toContain("server may reject");
    expect(wrapper.get("[data-test='generate-button']").attributes("disabled")).toBeUndefined();
  });

  it("lets a real blocker win over the advisory", () => {
    const wrapper = mountComposer(baseForm(), {
      disabled: true,
      disabledReason: "Use the reviewed variations panel.",
      warningReason: "The server may reject this size.",
    });
    const blocker = wrapper.get("[data-test='action-blocker']");
    expect(blocker.attributes("data-variant")).toBe("error");
    expect(blocker.text()).toContain("reviewed variations");
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

// The prompt rule's authority is the selected recipe, projected onto the form
// by `applyModelDefaults` — a recipe that IGNORES the prompt (no text encoder
// anywhere in the family) says so in the prompt bed rather than asking for a
// description the engine will never read.
describe("ComposerCard — prompt requirement", () => {
  it("names the note placeholder for a recipe that ignores the prompt", () => {
    const form = baseForm();
    form.family = "hunyuan3d";
    form.recipeCapabilities = recipeCapabilitiesSnapshot(hunyuan3dRecipe(), "hunyuan3d");
    const wrapper = mountComposer(form);
    expect(wrapper.get("textarea[aria-label='Prompt']").attributes("placeholder")).toBe(
      IGNORED_PROMPT_PLACEHOLDER,
    );
  });

  it("keeps this surface's own wording for a raster recipe", () => {
    const form = baseForm();
    form.recipeCapabilities = recipeCapabilitiesSnapshot(sdxlRecipe(), "sdxl");
    const wrapper = mountComposer(form);
    expect(wrapper.get("textarea[aria-label='Prompt']").attributes("placeholder")).toBe(
      "Describe the image you want to create…",
    );
  });
});

// Expand and Remix rewrite the prompt. A recipe that IGNORES it reads nothing
// they could produce, so the composer refuses both here and the ⌘E shortcut
// hands the intent to the view, which answers with the same sentence.
describe("ComposerCard — prompt transforms a recipe ignores", () => {
  function ignoredForm(): GenerateForm {
    const form = baseForm();
    form.family = "hunyuan3d";
    form.recipeCapabilities = recipeCapabilitiesSnapshot(hunyuan3dRecipe(), "hunyuan3d");
    return form;
  }

  it("disables Expand and Remix with the reason, even with a prompt typed", () => {
    const wrapper = mountComposer(ignoredForm());
    const control = wrapper.findComponent(ExpandControl);
    expect(control.props("transformBlockedReason")).toBe(PROMPT_IGNORED_TRANSFORM_REASON);
    expect(wrapper.get('[data-test="expand-action"]').attributes("disabled")).toBeDefined();
    expect(wrapper.get('[data-test="remix-action"]').attributes("disabled")).toBeDefined();
    expect(wrapper.get('[data-test="expand-action"]').attributes("title")).toBe(
      PROMPT_IGNORED_TRANSFORM_REASON,
    );
    expect(wrapper.get('[data-test="transform-blocked-hint"]').text()).toBe(
      PROMPT_IGNORED_TRANSFORM_REASON,
    );
  });

  it("still hands the ⌘E shortcut to the view so it can answer with the reason", async () => {
    const wrapper = mountComposer(ignoredForm());
    await wrapper
      .get("textarea[aria-label='Prompt']")
      .trigger("keydown", { key: "e", metaKey: true });
    expect(wrapper.emitted("expand")).toHaveLength(1);
  });

  it("leaves both transforms available for a recipe that reads the prompt", () => {
    const form = baseForm();
    form.recipeCapabilities = recipeCapabilitiesSnapshot(sdxlRecipe(), "sdxl");
    const wrapper = mountComposer(form);
    expect(wrapper.findComponent(ExpandControl).props("transformBlockedReason")).toBeNull();
    expect(wrapper.get('[data-test="expand-action"]').attributes("disabled")).toBeUndefined();
    expect(wrapper.find('[data-test="transform-blocked-hint"]').exists()).toBe(false);
  });
});
