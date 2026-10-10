import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import { nextTick } from "vue";
import {
  createOverlayToken,
  pushOverlay,
  popOverlay,
} from "@ui/lib/overlayStack";
import PromptEditor from "./PromptEditor.vue";

const mountEditor = () =>
  mount(PromptEditor, {
    props: {
      open: true,
      prompt: "first\nsecond",
      history: ["recent fox", "older bird"],
    },
    global: { stubs: { teleport: true } },
  });
describe("PromptEditor", () => {
  it("authors live multiline text without intercepting arrows or newline", async () => {
    const w = mountEditor();
    const text = w.get("textarea");
    await text.setValue("long\nnew prompt");
    expect(w.emitted("authored")?.at(-1)).toEqual([
      "long\nnew prompt",
      "typed",
    ]);
    const arrow = new KeyboardEvent("keydown", {
      key: "ArrowUp",
      cancelable: true,
      bubbles: true,
    });
    text.element.dispatchEvent(arrow);
    expect(arrow.defaultPrevented).toBe(false);
    const chord = new KeyboardEvent("keydown", {
      key: "Enter",
      metaKey: true,
      cancelable: true,
      bubbles: true,
    });
    text.element.dispatchEvent(chord);
    expect(chord.defaultPrevented).toBe(true);
    w.unmount();
  });
  it("clears live text, undoes clear, and retires undo after an edit", async () => {
    const w = mountEditor();
    await w.get('[data-test="prompt-clear"]').trigger("click");
    expect(w.emitted("authored")?.at(-1)).toEqual(["", "clear"]);
    await w.setProps({ prompt: "" });
    await w.get('[data-test="prompt-undo-clear"]').trigger("click");
    expect(w.emitted("authored")?.at(-1)).toEqual([
      "first\nsecond",
      "undo-clear",
    ]);
    await w.setProps({ prompt: "first\nsecond" });
    await w.get('[data-test="prompt-clear"]').trigger("click");
    await w.setProps({ prompt: "" });
    await w.get("textarea").setValue("fresh");
    expect(w.find('[data-test="prompt-undo-clear"]').exists()).toBe(false);
    w.unmount();
  });
  it("searches recent prompts and recalls text only; Done only dismisses", async () => {
    const w = mountEditor();
    await w.get('[data-test="prompt-recent-toggle"]').trigger("click");
    await w.get('input[type="search"]').setValue("bird");
    const results = w.findAll('[data-test="prompt-history-item"]');
    expect(results).toHaveLength(1);
    await results[0]!.trigger("click");
    expect(w.emitted("authored")?.at(-1)).toEqual(["older bird", "recalled"]);
    await w.get('[data-test="prompt-done"]').trigger("click");
    expect(w.emitted("close")).toHaveLength(1);
    w.unmount();
  });
});

it("locks background scroll, restores editor after nested review and opener on close", async () => {
  const opener = document.createElement("button");
  document.body.append(opener);
  opener.focus();
  const w = mount(PromptEditor, {
    attachTo: document.body,
    props: { open: true, prompt: "draft" },
  });
  await nextTick();
  await nextTick();
  const text = document.querySelector('textarea[aria-label="Prompt text"]');
  expect(document.activeElement).toBe(text);
  expect(document.body.style.overflow).toBe("hidden");
  const review = createOverlayToken();
  pushOverlay(review);
  opener.focus();
  popOverlay(review);
  await nextTick();
  expect(document.activeElement).toBe(text);
  await w.setProps({ open: false });
  await nextTick();
  expect(document.activeElement).toBe(opener);
  expect(document.body.style.overflow).not.toBe("hidden");
  w.unmount();
  opener.remove();
});

it("reopens at the editable prompt after closing Recent, without old Clear recovery", async () => {
  const w = mount(PromptEditor, {
    attachTo: document.body,
    props: { open: true, prompt: "draft", history: ["older"] },
  });
  await nextTick();
  await nextTick();
  (
    document.querySelector('[data-test="prompt-clear"]') as HTMLButtonElement
  ).click();
  await w.setProps({ prompt: "" });
  (
    document.querySelector(
      '[data-test="prompt-recent-toggle"]',
    ) as HTMLButtonElement
  ).click();
  await nextTick();
  await w.setProps({ open: false });
  await w.setProps({ open: true });
  await nextTick();
  await nextTick();
  expect(document.querySelector('input[type="search"]')).toBeNull();
  expect(document.querySelector('[data-test="prompt-undo-clear"]')).toBeNull();
  expect(document.activeElement).toBe(
    document.querySelector('textarea[aria-label="Prompt text"]'),
  );
  w.unmount();
});
it("tracks the panned visual viewport while keyboard space changes", async () => {
  const original = Object.getOwnPropertyDescriptor(window, "visualViewport");
  const viewport = Object.assign(new EventTarget(), {
    height: 350,
    offsetTop: 90,
  });
  Object.defineProperty(window, "visualViewport", {
    configurable: true,
    value: viewport,
  });
  const w = mount(PromptEditor, {
    attachTo: document.body,
    props: { open: true, prompt: "draft" },
  });
  await nextTick();
  const dialog = document.querySelector<HTMLElement>(".prompt-editor")!;
  expect(dialog.style.getPropertyValue("--prompt-editor-height")).toBe("350px");
  expect(dialog.style.getPropertyValue("--prompt-editor-top")).toBe("90px");
  viewport.height = 280;
  viewport.offsetTop = 120;
  viewport.dispatchEvent(new Event("scroll"));
  await nextTick();
  expect(dialog.style.getPropertyValue("--prompt-editor-top")).toBe("120px");
  expect(dialog.style.getPropertyValue("--prompt-editor-height")).toBe("280px");
  w.unmount();
  if (original) Object.defineProperty(window, "visualViewport", original);
  else
    delete (window as unknown as { visualViewport?: unknown }).visualViewport;
});
