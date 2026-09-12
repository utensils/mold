import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import ComposerCard from "./ComposerCard.vue";
import composerSource from "./ComposerCard.vue?raw";
import { PROMPT_IGNORED_TRANSFORM_REASON } from "@studio/lib/promptTransform";

/*
 * The two composer chords are the platform's own: ⌘ on Apple, Ctrl elsewhere.
 * On macOS Ctrl+E is move-to-end-of-line, so accepting either modifier took a
 * key the system already owns. The suite drives the real conventions through a
 * platform this test chooses, so both the gate and the keycap are pinned.
 */
const platform = vi.hoisted(() => ({ current: "macos" as "macos" | "linux" }));
vi.mock("../../lib/platform", async () => {
  const studio = await vi.importActual<typeof import("@studio/lib/platform")>(
    "@studio/lib/platform",
  );
  return {
    primaryModifierPressed: (event: KeyboardEvent) =>
      studio.primaryModifierPressed(event, platform.current),
    shortcutLabel: (key: string) =>
      `${studio.platformUi(platform.current).modifierLabel}${key}`,
  };
});

afterEach(() => {
  platform.current = "macos";
});

function factory(
  props: Partial<InstanceType<typeof ComposerCard>["$props"]> & {
    slots?: Record<string, string>;
  } = {},
) {
  const { slots, ...rest } = props;
  return mount(ComposerCard, {
    props: {
      prompt: "a lighthouse",
      aspectLabel: "1:1",
      width: 1024,
      height: 1024,
      steps: 28,
      batchSize: 1,
      ...rest,
    },
    ...(slots ? { slots } : {}),
  });
}

describe("ComposerCard", () => {
  it("submits on ⌘↵ on a Mac, and refuses the Ctrl chord the system owns", async () => {
    const wrapper = factory();
    const ta = wrapper.get("[data-test='composer-prompt']");
    await ta.trigger("keydown", { key: "Enter", metaKey: true });
    expect(wrapper.emitted("submit")).toHaveLength(1);

    await ta.trigger("keydown", { key: "Enter", ctrlKey: true });
    expect(wrapper.emitted("submit")).toHaveLength(1);
    expect(wrapper.get("[data-test='composer-submit']").text()).toContain("⌘↵");
  });

  it("submits on Ctrl+↵ off a Mac, and refuses the Command chord", async () => {
    platform.current = "linux";
    const wrapper = factory();
    const ta = wrapper.get("[data-test='composer-prompt']");
    await ta.trigger("keydown", { key: "Enter", ctrlKey: true });
    expect(wrapper.emitted("submit")).toHaveLength(1);

    await ta.trigger("keydown", { key: "Enter", metaKey: true });
    expect(wrapper.emitted("submit")).toHaveLength(1);
    expect(wrapper.get("[data-test='composer-submit']").text()).toContain(
      "Ctrl+↵",
    );
  });

  it("does not submit on a plain Enter", async () => {
    const wrapper = factory();
    await wrapper
      .get("[data-test='composer-prompt']")
      .trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("submit")).toBeUndefined();
  });

  it("tags a ↑/↓ history recall so the page can release a quick expansion", async () => {
    const wrapper = factory({ prompt: "storm light", history: ["newest"] });
    const ta = wrapper.get("[data-test='composer-prompt']");
    const el = ta.element as HTMLTextAreaElement;
    el.setSelectionRange(0, 0);
    await ta.trigger("keydown", { key: "ArrowUp" });
    expect(wrapper.emitted("update:prompt")?.at(-1)).toEqual([
      "newest",
      "recalled",
    ]);
    el.setSelectionRange(el.value.length, el.value.length);
    await ta.trigger("keydown", { key: "ArrowDown" });
    expect(wrapper.emitted("update:prompt")?.at(-1)).toEqual([
      "storm light",
      "recalled",
    ]);
  });

  it("tags hand edits as typing so a quick expansion keeps its stale recovery", async () => {
    const wrapper = factory();
    await wrapper.get("[data-test='composer-prompt']").setValue("edited");
    expect(wrapper.emitted("update:prompt")?.at(-1)).toEqual([
      "edited",
      "typed",
    ]);
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

  it("shows an actionable prerequisite and disables Generate", async () => {
    const wrapper = factory({
      disabledReason:
        "This reviewed MiniMax H3 runtime requires a first frame.",
    });

    expect(wrapper.text()).toContain("requires a first frame");
    expect(
      wrapper.get("[data-test='composer-submit']").attributes("disabled"),
    ).toBeDefined();
    await wrapper
      .get("[data-test='composer-prompt']")
      .trigger("keydown", { key: "Enter", metaKey: true });
    expect(wrapper.emitted("submit")).toBeUndefined();
  });

  it("emits the prompt on input, tagged with how it arrived", async () => {
    const wrapper = factory();
    const ta = wrapper.get("[data-test='composer-prompt']")
      .element as HTMLTextAreaElement;
    ta.value = "a lighthouse in a storm";
    await wrapper.get("[data-test='composer-prompt']").trigger("input");
    expect(wrapper.emitted("update:prompt")?.[0]).toEqual([
      "a lighthouse in a storm",
      "typed",
    ]);
  });

  /*
   * Style is the style — the checkpoint the picture is made with. The prompt
   * presets that also called themselves Style put a second "Photoreal" on the
   * same screen meaning something else entirely, so the strip is retired here
   * as it already is on the app and the phone.
   */
  it("offers no prompt-preset strip beside the style picker", () => {
    const wrapper = factory();
    expect(wrapper.find("[data-test='style-toggle']").exists()).toBe(false);
    expect(wrapper.find("[data-test='style-chips']").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("Cinematic");
  });

  it("renders the summary line, adding ×N only for a batch", () => {
    expect(factory().get("[data-test='composer-summary']").text()).toBe(
      "1:1 · 1024×1024 · 28 passes",
    );
    const batched = factory({ batchSize: 3 });
    expect(batched.get("[data-test='composer-summary']").text()).toBe(
      "1:1 · 1024×1024 · 28 passes · ×3",
    );
  });

  it("asks for more words in the app's own words, for the current batch size", () => {
    expect(factory().get("[data-test='composer-expand']").text()).toContain(
      "Write more for me",
    );
    expect(
      factory({ batchSize: 4 }).get("[data-test='composer-expand']").text(),
    ).toContain("Write 4 for me");
  });

  it("keeps Generate visible for multi-image batches", () => {
    const wrapper = factory({ batchSize: 4 });
    expect(wrapper.get("[data-test='composer-submit']").text()).toContain(
      "Generate",
    );
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

  // A recipe that IGNORES the prompt (Hunyuan3D has no text encoder anywhere)
  // gives a rewrite nothing to act on, so both transforms are refused with the
  // one shared reason instead of sending a request the host answers with advice.
  it("disables Expand and Remix with the reason when the recipe ignores the prompt", () => {
    const wrapper = factory({
      transformBlockedReason: PROMPT_IGNORED_TRANSFORM_REASON,
    });
    const expand = wrapper.get("[data-test='composer-expand']");
    const remix = wrapper.get("[data-test='composer-remix']");
    expect(expand.attributes("disabled")).toBeDefined();
    expect(remix.attributes("disabled")).toBeDefined();
    expect(expand.attributes("title")).toBe(PROMPT_IGNORED_TRANSFORM_REASON);
    expect(remix.attributes("title")).toBe(PROMPT_IGNORED_TRANSFORM_REASON);
    expect(wrapper.get("[data-test='composer-transform-blocked']").text()).toBe(
      PROMPT_IGNORED_TRANSFORM_REASON,
    );
    // Generating is still the whole point of the surface — only the prompt
    // transforms are unavailable.
    expect(
      wrapper.get("[data-test='composer-submit']").attributes("disabled"),
    ).toBeUndefined();
  });

  it("keeps Expand and Remix live when nothing blocks the transforms", () => {
    const wrapper = factory();
    expect(
      wrapper.get("[data-test='composer-expand']").attributes("disabled"),
    ).toBeUndefined();
    expect(
      wrapper.get("[data-test='composer-remix']").attributes("disabled"),
    ).toBeUndefined();
    expect(
      wrapper.find("[data-test='composer-transform-blocked']").exists(),
    ).toBe(false);
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

/*
 * The mock's composer is the whole control surface: one row of chips left of
 * the prompt transforms, Generate at its right end, and nothing about the
 * picture anywhere else on a phone. The chips are filled by the page, because
 * the style picker, the shape resolver and the batch count all belong to the
 * form the page owns — the composer only decides where they sit.
 */
describe("ComposerCard action row", () => {
  it("renders Style, Shape and Make, in that order, before Write more for me", () => {
    const wrapper = factory({
      slots: {
        style: "<span data-test='slot-style'>Photoreal</span>",
        shape: "<span data-test='slot-shape'>Square · 1024</span>",
        count: "<span data-test='slot-count'>Make 4</span>",
      },
    });
    const order = [
      "[data-test='slot-style']",
      "[data-test='slot-shape']",
      "[data-test='slot-count']",
      "[data-test='composer-expand']",
      "[data-test='composer-submit']",
    ].map((probe) => {
      const el = wrapper.get(probe).element;
      return [...wrapper.element.querySelectorAll("*")].indexOf(el);
    });
    expect(order).toEqual([...order].sort((a, b) => a - b));
    expect(order[0]).toBeGreaterThan(-1);
  });

  it("keeps the three chips on their own row, with the summary at its end", () => {
    // Measured on plato at 1440px: with the summary leading the one action
    // row, the style chip pushed Shape and Make onto a second line and
    // Generate onto a third. The chips are one row; the transforms and
    // Generate are the next.
    const wrapper = factory({
      slots: {
        style: "<span data-test='slot-style'>Photoreal</span>",
        shape: "<span data-test='slot-shape'>Square · 1024</span>",
        count: "<span data-test='slot-count'>Make 4</span>",
      },
    });
    const chips = wrapper.get("[data-test='composer-chips']");
    for (const probe of ["slot-style", "slot-shape", "slot-count"]) {
      expect(chips.find(`[data-test='${probe}']`).exists()).toBe(true);
    }
    expect(chips.find("[data-test='composer-summary']").exists()).toBe(true);
    expect(chips.find("[data-test='composer-submit']").exists()).toBe(false);
    const actions = wrapper.get("[data-test='composer-actions']");
    expect(actions.find("[data-test='composer-expand']").exists()).toBe(true);
    expect(actions.find("[data-test='composer-submit']").exists()).toBe(true);
    expect(actions.find("[data-test='slot-style']").exists()).toBe(false);
  });

  it("renders none of the three when the page fills none of them", () => {
    const wrapper = factory();
    expect(wrapper.find("[data-test='slot-style']").exists()).toBe(false);
  });

  /*
   * The phone used to render the whole settings column INSIDE the composer,
   * which is why the narrow page was longer than the wide one. The rail is one
   * sheet now, so the composer has no phone-only well to fill.
   */
  it("has no phone-only controls well", () => {
    const wrapper = factory({
      slots: { "mobile-controls": "<span data-test='legacy-mobile'>x</span>" },
    });
    expect(wrapper.find("[data-test='legacy-mobile']").exists()).toBe(false);
    expect(composerSource).not.toContain("mobile-controls");
  });

  /*
   * The chip carries the keycap, so the keycap has to be true — desktop's
   * ⌘E reaches the same rewrite from inside the prompt bed.
   */
  it("rewrites the prompt on ⌘E, and refuses when the recipe reads no prompt", async () => {
    const wrapper = factory();
    const bed = wrapper.get("[data-test='composer-prompt']");
    await bed.trigger("keydown", { key: "e", metaKey: true });
    expect(wrapper.emitted("expand")).toHaveLength(1);
    expect(wrapper.get("[data-test='composer-expand']").text()).toContain("⌘E");

    const blocked = factory({ transformBlockedReason: "No text encoder." });
    await blocked
      .get("[data-test='composer-prompt']")
      .trigger("keydown", { key: "e", metaKey: true });
    expect(blocked.emitted("expand")).toBeUndefined();
  });

  /*
   * Ctrl+E is move-to-end-of-line on a Mac, so the rewrite chord may not take
   * it there — and the keycap may not promise ⌘ on a machine with no ⌘ key.
   */
  it("leaves Ctrl+E to macOS, and spells the rewrite chord Ctrl+E off a Mac", async () => {
    const mac = factory();
    await mac
      .get("[data-test='composer-prompt']")
      .trigger("keydown", { key: "e", ctrlKey: true });
    expect(mac.emitted("expand")).toBeUndefined();

    platform.current = "linux";
    const linux = factory();
    const bed = linux.get("[data-test='composer-prompt']");
    await bed.trigger("keydown", { key: "e", metaKey: true });
    expect(linux.emitted("expand")).toBeUndefined();
    await bed.trigger("keydown", { key: "E", ctrlKey: true });
    expect(linux.emitted("expand")).toHaveLength(1);
    expect(linux.get("[data-test='composer-expand']").text()).toContain(
      "Ctrl+E",
    );
  });

  /*
   * The kit has three control heights and Generate is the tallest of them.
   * It was specced at 42px, a fourth height nothing else on the screen uses.
   */
  it("stands Generate on the kit's own large control height", () => {
    expect(composerSource).toContain("height: var(--mold-ctl-lg, 32px)");
    expect(composerSource).not.toContain("height: 42px");
  });

  /*
   * Where the composer sits is the PAGE's decision — sticky inside the wide
   * column, fixed to the bottom of a narrow one — so the page owns BOTH
   * rules and the card hard-codes no position. The card once carried a
   * second copy of each with different z-indexes; both scoped rules landed
   * on the same element at equal specificity, so bundle order decided.
   */
  it("takes no position of its own and defines neither of the page's classes", () => {
    expect(composerSource).not.toContain(".composer--sticky");
    expect(composerSource).not.toContain(".composer--docked");
    const base = composerSource.match(/\n\.composer \{[^}]*\}/)?.[0] ?? "";
    expect(base).not.toContain("position:");
  });
});

/*
 * A rewrite runs on a MACHINE, so while it runs the composer says which one —
 * desktop's `ExpandControl` progress line, word for word, in a live region.
 * Web used to open a dialog with an "Expanding…" button instead, which said
 * nothing about where the work was happening.
 */
describe("ComposerCard — the rewrite's progress", () => {
  it("names the machine while a batch-1 rewrite runs, in a live region", () => {
    const wrapper = factory({
      running: true,
      expansionHostLabel: "Studio 4090",
    });
    const status = wrapper.get("[data-test='composer-expand-progress']");
    expect(status.text()).toBe("Writing more on Studio 4090…");
    expect(status.attributes("role")).toBe("status");
    expect(status.attributes("aria-live")).toBe("polite");
  });

  it("counts the versions while a prepared batch runs", () => {
    expect(
      factory({
        running: true,
        batchSize: 3,
        expansionHostLabel: "Studio 4090",
      })
        .get("[data-test='composer-expand-progress']")
        .text(),
    ).toBe("Writing 3 versions on Studio 4090…");
  });

  it("falls back to the selected machine when none is named yet", () => {
    expect(
      factory({ running: true })
        .get("[data-test='composer-expand-progress']")
        .text(),
    ).toBe("Writing more on the selected machine…");
  });

  it("says nothing and keeps both transforms live when nothing is running", () => {
    const wrapper = factory();
    expect(
      wrapper.find("[data-test='composer-expand-progress']").exists(),
    ).toBe(false);
    expect(
      wrapper.get("[data-test='composer-expand']").attributes("disabled"),
    ).toBeUndefined();
  });

  it("refuses a second rewrite — by click or by chord — while one is running", async () => {
    const wrapper = factory({ running: true });
    expect(
      wrapper.get("[data-test='composer-expand']").attributes("disabled"),
    ).toBe("");
    await wrapper
      .get("[data-test='composer-prompt']")
      .trigger("keydown", { key: "e", metaKey: true });
    expect(wrapper.emitted("expand")).toBeUndefined();
  });
});
