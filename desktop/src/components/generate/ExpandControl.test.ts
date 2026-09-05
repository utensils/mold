import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ExpandControl from "./ExpandControl.vue";
import controlSource from "./ExpandControl.vue?raw";
import { PROMPT_IGNORED_TRANSFORM_REASON } from "@studio/lib/promptTransform";

const wrappers: ReturnType<typeof mount>[] = [];

function mountControl(
  props: Partial<{
    prompt: string;
    batchSize: number;
    running: boolean;
    hostLabel: string | null;
    canUndo: boolean;
    transformBlockedReason: string | null;
    originalAvailable: boolean;
    remixSource: "original" | "current";
  }> = {},
) {
  const wrapper = mount(ExpandControl, {
    props: {
      prompt: "a cat",
      batchSize: 1,
      running: false,
      hostLabel: "Studio 4090",
      canUndo: false,
      ...props,
    },
  });
  wrappers.push(wrapper);
  return wrapper;
}

/** Remix, its source and the undo live behind the chip's caret. */
async function openMore(wrapper: ReturnType<typeof mount>) {
  await wrapper.get('[data-test="rewrite-more"]').trigger("click");
  return wrapper;
}

afterEach(() => {
  while (wrappers.length) wrappers.pop()!.unmount();
});

describe("ExpandControl", () => {
  it("preserves the Batch 1 quick-expand and undo interaction", async () => {
    const wrapper = mountControl({ canUndo: true });
    await wrapper.get('button[title="Write more for me"]').trigger("click");
    expect(wrapper.emitted("expand")).toHaveLength(1);

    // Undo moved under the caret with the rest of the secondary actions; its
    // accessible name is unchanged.
    await openMore(wrapper);
    await wrapper.get('button[aria-label="Restore original prompt"]').trigger("click");
    expect(wrapper.emitted("restore")).toHaveLength(1);
  });

  it("turns Batch N into a preparation action", async () => {
    const wrapper = mountControl({ batchSize: 3 });
    const button = wrapper.get('[data-test="expand-action"]');
    expect(button.text()).toContain("Prepare 3 variations");
    await button.trigger("click");
    expect(wrapper.emitted("expand")).toHaveLength(1);
    await openMore(wrapper);
    expect(wrapper.find('[aria-label="Restore original prompt"]').exists()).toBe(false);
  });

  it("announces progress with the frozen host and requested count", () => {
    const wrapper = mountControl({ batchSize: 5, running: true });
    expect(wrapper.get('[role="status"]').text()).toBe("Writing 5 versions on Studio 4090…");
    expect(wrapper.get('[data-test="expand-action"]').attributes("disabled")).toBeDefined();
  });

  it("keeps the exposed keyboard action wired to the visible action", () => {
    const wrapper = mountControl({ batchSize: 3 });
    (wrapper.vm as unknown as { expand: () => void }).expand();
    expect(wrapper.emitted("expand")).toHaveLength(1);
  });
});

/*
 * The composer's control row is 28px chips. Two 26px toolbar buttons plus a
 * Source <select> sat on the same baseline and added ~140px to a row that
 * already wrapped — which dropped Generate onto a second line and made the
 * composer taller, taking that height from the canvas above it.
 */
describe("ExpandControl — one chip, not two buttons", () => {
  it("renders the mock's 28px chip with the sparkle and the chord", () => {
    const wrapper = mountControl();
    const chip = wrapper.get(".ms-rewrite");
    expect(chip.find("svg").exists()).toBe(true);
    expect(chip.text()).toContain("Write more for me");
    expect(wrapper.get(".ms-rewrite__chord").text()).toMatch(/E$/);
    expect(controlSource).toMatch(/\.ms-rewrite\s*\{[^}]*height:\s*28px/s);
    expect(controlSource).toContain('name="sparkle"');
    // The 26px toolbar button is gone from this control entirely.
    expect(controlSource).not.toContain("ms-toolbar-button");
  });

  it("hides Remix and its source behind the caret", () => {
    const wrapper = mountControl({ originalAvailable: true });
    expect(wrapper.find('[data-test="remix-action"]').exists()).toBe(false);
    expect(wrapper.find('[data-test="remix-source-original"]').exists()).toBe(false);
    expect(wrapper.find('[data-test="rewrite-more"]').exists()).toBe(true);
  });

  it("emits remix from the folded action and closes the menu", async () => {
    const wrapper = await openMore(mountControl());
    await wrapper.get('[data-test="remix-action"]').trigger("click");
    expect(wrapper.emitted("remix")).toHaveLength(1);
    expect(wrapper.find('[data-test="rewrite-menu"]').exists()).toBe(false);
  });

  /* Replaces the old `remix-source-select` <select>: the same two choices,
     as menu rows, emitting the same contract. */
  it("keeps both remix sources and marks the active one", async () => {
    const wrapper = await openMore(
      mountControl({ originalAvailable: true, remixSource: "original" }),
    );
    expect(wrapper.get('[data-test="remix-source-original"]').attributes("aria-checked")).toBe(
      "true",
    );
    expect(wrapper.get('[data-test="remix-source-current"]').attributes("aria-checked")).toBe(
      "false",
    );
    await wrapper.get('[data-test="remix-source-current"]').trigger("click");
    expect(wrapper.emitted("update:remixSource")).toEqual([["current"]]);
  });

  it("offers no source choice when there is no original to go back to", async () => {
    const wrapper = await openMore(mountControl({ originalAvailable: false }));
    expect(wrapper.find('[data-test="remix-source-original"]').exists()).toBe(false);
  });

  it("closes the menu on Escape", async () => {
    const wrapper = await openMore(mountControl());
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await wrapper.vm.$nextTick();
    expect(wrapper.find('[data-test="rewrite-menu"]').exists()).toBe(false);
  });
});

// A recipe that IGNORES the prompt has no text encoder to read a rewrite, so
// both transforms are refused here rather than sending a request the host
// answers with advice.
describe("ExpandControl — a recipe that ignores the prompt", () => {
  it("disables both transforms, naming the reason in the tooltip and a visible hint", async () => {
    const wrapper = await openMore(
      mountControl({
        batchSize: 3,
        transformBlockedReason: PROMPT_IGNORED_TRANSFORM_REASON,
      }),
    );
    const expand = wrapper.get('[data-test="expand-action"]');
    const remix = wrapper.get('[data-test="remix-action"]');
    expect(expand.attributes("disabled")).toBeDefined();
    expect(remix.attributes("disabled")).toBeDefined();
    expect(expand.attributes("title")).toBe(PROMPT_IGNORED_TRANSFORM_REASON);
    expect(remix.attributes("title")).toBe(PROMPT_IGNORED_TRANSFORM_REASON);
    expect(wrapper.get('[data-test="transform-blocked-hint"]').text()).toBe(
      PROMPT_IGNORED_TRANSFORM_REASON,
    );
  });

  it("no-ops the exposed keyboard action instead of asking for a rewrite", () => {
    const wrapper = mountControl({ transformBlockedReason: PROMPT_IGNORED_TRANSFORM_REASON });
    (wrapper.vm as unknown as { expand: () => void }).expand();
    expect(wrapper.emitted("expand")).toBeUndefined();
  });

  it("refuses the folded remix as well as the visible verb", async () => {
    const wrapper = await openMore(
      mountControl({ transformBlockedReason: PROMPT_IGNORED_TRANSFORM_REASON }),
    );
    await wrapper.get('[data-test="remix-action"]').trigger("click");
    expect(wrapper.emitted("remix")).toBeUndefined();
  });

  it("keeps both transforms available when nothing blocks them", async () => {
    const wrapper = await openMore(mountControl({ transformBlockedReason: null }));
    expect(wrapper.get('[data-test="expand-action"]').attributes("disabled")).toBeUndefined();
    expect(wrapper.get('[data-test="remix-action"]').attributes("disabled")).toBeUndefined();
    expect(wrapper.find('[data-test="transform-blocked-hint"]').exists()).toBe(false);
  });
});
