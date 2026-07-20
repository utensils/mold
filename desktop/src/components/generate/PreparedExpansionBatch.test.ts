import { afterEach, describe, expect, it } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import PreparedExpansionBatch from "./PreparedExpansionBatch.vue";
import { createPreparedExpansionBatch } from "../../lib/preparedExpansion";
import type { ExpansionPullView } from "../../lib/expansionPull";

const batch = () =>
  createPreparedExpansionBatch(
    {
      sourcePrompt: "source",
      model: "flux-dev:q8",
      family: "flux",
      requestedCount: 3,
      selectedHostPolicy: null,
    },
    {
      hostId: "studio",
      label: "Studio 4090",
      kind: "remote",
      target: { baseUrl: "http://studio:7680", apiKey: "secret" },
    },
    ["one", "two", "three"],
    1,
  );

const wrappers: ReturnType<typeof mount>[] = [];
function mountBatch(
  props: Partial<{
    batch: ReturnType<typeof batch>;
    staleReasons: string[];
    preparing: boolean;
    error: string | null;
    pullStatus: ExpansionPullView | null;
    pullModel: string | null;
    pullHostLabel: string | null;
    pullEtaSeconds: number | null;
    activeHostLabel: string | null;
    submitting: boolean;
  }> = {},
) {
  const wrapper = mount(PreparedExpansionBatch, {
    attachTo: document.body,
    props: {
      batch: batch(),
      staleReasons: [],
      preparing: false,
      error: null,
      pullStatus: null,
      pullModel: null,
      pullHostLabel: null,
      pullEtaSeconds: null,
      activeHostLabel: null,
      submitting: false,
      ...props,
    },
  });
  wrappers.push(wrapper);
  return wrapper;
}

afterEach(() => {
  while (wrappers.length) wrappers.pop()!.unmount();
});

describe("PreparedExpansionBatch", () => {
  it("renders editable, position-labelled prompts and host provenance", async () => {
    const wrapper = mountBatch();
    expect(wrapper.get('[data-test="prepared-expansion-batch"]').classes()).toContain(
      "max-h-[min(34rem,60vh)]",
    );
    expect(wrapper.text()).toContain("Studio 4090");
    expect(wrapper.text()).toContain("Auto resolved");
    const editors = wrapper.findAll("textarea");
    expect(editors).toHaveLength(3);
    expect(editors[0]!.attributes("aria-label")).toBe("Variation 1 of 3");

    await editors[0]!.setValue("edited one");
    expect(wrapper.emitted("edit")).toEqual([[{ id: "prepared-1-1", text: "edited one" }]]);
  });

  it("removes a prompt directly when more than two remain", async () => {
    const wrapper = mountBatch();
    await wrapper.get('[aria-label="Remove variation 2"]').trigger("click");
    expect(wrapper.emitted("remove")).toEqual([["prepared-1-2"]]);
  });

  it("requires explicit inline confirmation before reducing to ordinary Batch 1", async () => {
    const two = batch();
    two.prompts.pop();
    two.requestedCount = 2;
    const wrapper = mountBatch({ batch: two });

    await wrapper.get('[aria-label="Remove variation 1"]').trigger("click");
    expect(wrapper.emitted("collapse")).toBeUndefined();
    expect(wrapper.get('[role="alertdialog"]').text()).toContain("Continue with one prompt?");

    await wrapper.get('[data-test="confirm-collapse"]').trigger("click");
    expect(wrapper.emitted("collapse")).toEqual([["prepared-1-1"]]);
  });

  it("returns focus to Remove when Keep prepared batch closes confirmation", async () => {
    const two = batch();
    two.prompts.pop();
    two.requestedCount = 2;
    const wrapper = mountBatch({ batch: two });
    const remove = wrapper.get('[aria-label="Remove variation 1"]');
    await remove.trigger("click");
    await wrapper.get('[data-test="keep-prepared-batch"]').trigger("click");
    expect(document.activeElement).toBe(remove.element);
  });

  it("returns focus to Remove when Escape closes confirmation", async () => {
    const two = batch();
    two.prompts.pop();
    two.requestedCount = 2;
    const wrapper = mountBatch({ batch: two });
    const remove = wrapper.get('[aria-label="Remove variation 1"]');
    await remove.trigger("click");
    await wrapper.get('[role="alertdialog"]').trigger("keydown", { key: "Escape" });
    expect(document.activeElement).toBe(remove.element);
  });

  it("returns focus to the first replacement editor after refresh", async () => {
    const stale = batch();
    const wrapper = mountBatch({ staleReasons: ["Batch changed from 3 to 4."] });
    const refresh = wrapper.get('[data-test="refresh-prepared"]');
    (refresh.element as HTMLButtonElement).focus();
    await refresh.trigger("click");
    const replacement = batch();
    replacement.batchId = "replacement";
    replacement.prompts = replacement.prompts.map((prompt, index) => ({
      ...prompt,
      id: `replacement-${index + 1}`,
    }));
    await wrapper.setProps({ batch: replacement, staleReasons: [] });
    await flushPromises();
    expect(document.activeElement).toBe(wrapper.findAll("textarea")[0]!.element);
    expect(stale.batchId).not.toBe(replacement.batchId);
  });

  it("does not steal focus when the user leaves during a delayed replacement", async () => {
    const wrapper = mountBatch({ staleReasons: ["Batch changed from 3 to 4."] });
    const refresh = wrapper.get('[data-test="refresh-prepared"]');
    (refresh.element as HTMLButtonElement).focus();
    await refresh.trigger("click");
    const outside = document.createElement("button");
    document.body.append(outside);
    outside.focus();

    const replacement = batch();
    replacement.batchId = "replacement-outside-focus";
    await wrapper.setProps({ batch: replacement, staleReasons: [] });
    await flushPromises();

    expect(document.activeElement).toBe(outside);
    outside.remove();
  });

  it("locks edits and removal while submitting but leaves discard available to cancel", () => {
    const wrapper = mountBatch({ submitting: true });
    expect(
      wrapper.findAll("textarea").every((editor) => editor.attributes("disabled") !== undefined),
    ).toBe(true);
    expect(wrapper.get('[aria-label="Remove variation 1"]').attributes("disabled")).toBeDefined();
    expect(wrapper.get('[data-test="discard-prepared"]').attributes("disabled")).toBeUndefined();
  });

  it("preserves stale work, names the reasons, and blocks generation", () => {
    const wrapper = mountBatch({
      staleReasons: [
        "Source prompt changed after these variations were prepared.",
        "Studio 4090 is no longer reachable.",
      ],
    });
    expect(wrapper.get('[role="alert"]').text()).toContain("Source prompt changed");
    expect(wrapper.get('[role="alert"]').text()).toContain("Studio 4090 is no longer reachable");
    expect(wrapper.get('[data-test="generate-prepared"]').attributes("disabled")).toBeDefined();
    expect(wrapper.get('[data-test="refresh-prepared"]')).toBeDefined();
  });

  it("blocks empty edited prompts without erasing them", () => {
    const invalid = batch();
    invalid.prompts[1]!.text = "   ";
    const wrapper = mountBatch({ batch: invalid });
    expect(wrapper.findAll('[aria-invalid="true"]')).toHaveLength(1);
    expect(wrapper.text()).toContain("Variation 2 needs a prompt");
    expect(wrapper.get('[data-test="generate-prepared"]').attributes("disabled")).toBeDefined();
  });

  it("exposes regenerate, discard, pull, and generate actions with explicit labels", async () => {
    const wrapper = mountBatch({
      error: "Expansion model is missing.",
      pullStatus: { kind: "missing", job: null },
      pullModel: "Qwen/Qwen3",
      pullHostLabel: "Studio 4090",
    });
    await wrapper.get('[data-test="regenerate-prepared"]').trigger("click");
    await wrapper.get('[data-test="discard-prepared"]').trigger("click");
    await wrapper.get('[data-test="pull-expand-model"]').trigger("click");
    await wrapper.get('[data-test="generate-prepared"]').trigger("click");
    expect(wrapper.emitted("regenerate")).toHaveLength(1);
    expect(wrapper.emitted("discard")).toHaveLength(1);
    expect(wrapper.emitted("pull")).toHaveLength(1);
    expect(wrapper.emitted("generate")).toHaveLength(1);
  });
});
