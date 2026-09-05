import { afterEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { reactive } from "vue";
import ComposerCard from "./ComposerCard.vue";
import ExpandControl from "../generate/ExpandControl.vue";
import Stepper from "@ui/components/Stepper.vue";
import { MAX_BATCH_SIZE, newGenerateForm, type GenerateForm } from "../../lib/generateForm";
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

function mountComposer(
  form: GenerateForm,
  overrides: Record<string, unknown> = {},
  slots: Record<string, string> = {},
) {
  return mount(ComposerCard, {
    attachTo: document.body,
    global: { stubs: { EstimateBadge: true } },
    slots,
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

  it("keeps a restored look preset on the form with no strip to show it", async () => {
    // The preset strip is gone from the desktop composer — its word collided
    // with the bound "Style" — and nothing here reads one any more. A
    // persisted draft's preset rides along untouched and changes no request;
    // the field survives for the phone, which still has its chips.
    const form = baseForm();
    form.stylePreset = "cinematic";
    const wrapper = mountComposer(form);
    await wrapper.vm.$nextTick();
    expect(wrapper.find("[data-test='style-toggle']").exists()).toBe(false);
    expect(form.stylePreset).toBe("cinematic");
    expect(form.prompt).toBe("a lighthouse");
  });

  it("says Generate in one word, and Cancel while a submission is in flight", () => {
    // The mono shortcut rides the button; the word beside it is the label.
    expect(mountComposer(baseForm()).get("[data-test='generate-button']").text()).toContain(
      "Generate",
    );
    expect(
      mountComposer(baseForm(), { submitting: true, buttonLabel: "Cancel" })
        .get("[data-test='generate-button']")
        .text(),
    ).toContain("Cancel");
  });

  it("puts the queue depth beside the button, never inside it", () => {
    const wrapper = mountComposer(baseForm(), { queuedNote: "+3 queued" });
    expect(wrapper.get("[data-test='generate-queued-note']").text()).toBe("+3 queued");
    expect(wrapper.get("[data-test='generate-button']").text()).not.toContain("queued");
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
      "Describe the picture you want — “a brass teapot on a rainy windowsill, evening light”",
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

  it("disables Expand and Remix with the reason, even with a prompt typed", async () => {
    const wrapper = mountComposer(ignoredForm());
    const control = wrapper.findComponent(ExpandControl);
    expect(control.props("transformBlockedReason")).toBe(PROMPT_IGNORED_TRANSFORM_REASON);
    expect(wrapper.get('[data-test="expand-action"]').attributes("disabled")).toBeDefined();
    // Remix folded under the rewrite chip's caret; it is still refused, and
    // still says why.
    await wrapper.get('[data-test="rewrite-more"]').trigger("click");
    expect(wrapper.get('[data-test="remix-action"]').attributes("disabled")).toBeDefined();
    expect(wrapper.get('[data-test="remix-action"]').attributes("title")).toBe(
      PROMPT_IGNORED_TRANSFORM_REASON,
    );
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

// "Make N" moved out of the inspector onto the composer's control row, where
// the count sits beside Generate. The Stepper contract is unchanged.
describe("ComposerCard — batch", () => {
  it("steps the batch size through the Stepper", async () => {
    const form = baseForm();
    const wrapper = mountComposer(form);
    const stepper = wrapper.get("[data-test='batch-chip']").findComponent(Stepper);
    stepper.vm.$emit("update:modelValue", 3);
    await wrapper.vm.$nextTick();
    expect(form.batchSize).toBe(3);
    expect(stepper.props("max")).toBe(MAX_BATCH_SIZE);
    expect(stepper.props("editable")).toBe(true);
  });

  it("nests a compact Stepper so the chip stays one bordered control", () => {
    const wrapper = mountComposer(baseForm());
    const stepper = wrapper.get("[data-test='batch-chip']").findComponent(Stepper);
    expect(stepper.props("compact")).toBe(true);
  });

  it("accepts a directly entered large positive batch", async () => {
    const form = baseForm();
    const wrapper = mountComposer(form);
    const input = wrapper.get('input[aria-label="How many to make"]');
    await input.setValue("1000");
    await input.trigger("change");
    await wrapper.vm.$nextTick();
    expect(form.batchSize).toBe(1000);
  });

  it("does not overwrite an uncommitted direct entry on an arrow key", async () => {
    const form = baseForm();
    const wrapper = mountComposer(form);
    const input = wrapper.get('input[aria-label="How many to make"]');
    (input.element as HTMLInputElement).value = "120";
    await input.trigger("keydown", { key: "ArrowUp" });
    expect(form.batchSize).toBe(1);
    expect((input.element as HTMLInputElement).value).toBe("120");
  });

  it("locks the batch to one for a recipe that renders one at a time", () => {
    // The edit-model rule itself lives on the view (`forcesBatchSizeOne`);
    // here the locked chip must read 1 and refuse both entry and stepping.
    const form = baseForm();
    form.batchSize = 4;
    const wrapper = mountComposer(form, { batchLocked: true });
    const stepper = wrapper.get("[data-test='batch-chip']").findComponent(Stepper);
    expect(stepper.props("modelValue")).toBe(1);
    expect(stepper.props("max")).toBe(1);
    expect(stepper.props("editable")).toBe(false);
    expect(wrapper.find('input[aria-label="How many to make"]').exists()).toBe(false);
  });
});

// The Shape chip carries the canvas summary the Create header used to print
// as "1:1 · 1024×1024 · N steps". A canvasless (3-D) recipe renders no pixel
// canvas — width/height sit at the recipe's zero default — so it shows no
// chip at all rather than a nonsensical "0×0".
describe("ComposerCard — shape and style chips", () => {
  it("names a square canvas once and a rectangular one by family", () => {
    const square = baseForm();
    square.width = 1024;
    square.height = 1024;
    expect(mountComposer(square).get("[data-test='shape-chip']").text()).toContain("Square · 1024");

    const wide = baseForm();
    wide.width = 1216;
    wide.height = 704;
    expect(mountComposer(wide).get("[data-test='shape-chip']").text()).toContain("16:9 · 1216×704");
  });

  it("omits the Shape chip for a canvasless (3-D mesh) recipe", () => {
    const form = baseForm();
    form.family = "hunyuan3d";
    form.width = 0;
    form.height = 0;
    form.recipeCapabilities = recipeCapabilitiesSnapshot(hunyuan3dRecipe(), "hunyuan3d");
    const wrapper = mountComposer(form);
    expect(wrapper.find("[data-test='shape-chip']").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("0×0");
  });

  it("opens the inspector's settings from the Shape chip", async () => {
    const wrapper = mountComposer(baseForm());
    await wrapper.get("[data-test='shape-chip']").trigger("click");
    expect(wrapper.emitted("open-shape")).toHaveLength(1);
  });

  /*
   * Style is no longer a door. Its chip IS the picker (StylePicker.vue), so
   * the composer stops carrying a `styleLabel`/`styleId` pair and an
   * `open-style` emit and takes the whole control through a slot — one
   * selector, opened where the user is looking. The chip's own behaviour is
   * covered by StylePicker.test.ts; what this component owes is the seat.
   */
  it("seats the style picker in the control row rather than a door of its own", () => {
    const wrapper = mountComposer(baseForm(), undefined, {
      style: '<button data-test="fake-style-chip">Style</button>',
    });
    const controls = wrapper.get(".ms-composer__controls");
    expect(controls.find("[data-test='fake-style-chip']").exists()).toBe(true);
    // The seat comes first, ahead of Shape — the mock's control row order.
    const order = [...controls.element.children].map((el) => el.getAttribute("data-test"));
    expect(order.indexOf("fake-style-chip")).toBeLessThan(order.indexOf("shape-chip"));
    expect(wrapper.emitted("open-style")).toBeUndefined();
  });
});

// Clip mode keeps this one composer and hands it the selected scene's words:
// the form's own prompt must stay untouched, and Generate still answers ⌘↩.
describe("ComposerCard — clip mode", () => {
  function clipMode(form: GenerateForm) {
    return mountComposer(form, {
      promptValue: "rain picks up",
      placeholder: "Scene 2 — describe what happens next",
      countLabel: "Make 1 clip",
      showExpand: false,
    });
  }

  it("carries the scene's words instead of the form's prompt", async () => {
    const form = baseForm();
    form.prompt = "a brass teapot";
    const wrapper = clipMode(form);
    const textarea = wrapper.get<HTMLTextAreaElement>("textarea[aria-label='Prompt']");

    expect(textarea.element.value).toBe("rain picks up");
    expect(textarea.attributes("placeholder")).toBe("Scene 2 — describe what happens next");

    await textarea.setValue("the camera drifts left");
    expect(wrapper.emitted("update:promptValue")?.at(-1)).toEqual(["the camera drifts left"]);
    expect(form.prompt).toBe("a brass teapot");
  });

  it("counts one clip and offers no rewrite of a scene", () => {
    const wrapper = clipMode(baseForm());
    expect(wrapper.get("[data-test='batch-chip']").text()).toBe("Make 1 clip");
    expect(wrapper.findComponent(ExpandControl).exists()).toBe(false);
  });

  it("still generates on ⌘↵", async () => {
    const wrapper = clipMode(baseForm());
    await wrapper
      .get("textarea[aria-label='Prompt']")
      .trigger("keydown", { key: "Enter", metaKey: true });
    expect(wrapper.emitted("generate")).toHaveLength(1);
  });
});
