import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import QualityLadder from "./QualityLadder.vue";
import { qualityPresets } from "@studio/lib/qualityPresets";
import type { IntegerControl } from "@studio/lib/generated/generationProfileV1";

const STEPS: IntegerControl = {
  default: 28,
  min: 8,
  max: 50,
  step: 1,
  mode: "adjustable",
  recommended: [8, 28, 50],
};

function factory(
  props: Partial<InstanceType<typeof QualityLadder>["$props"]> = {},
) {
  return mount(QualityLadder, {
    props: { presets: qualityPresets(STEPS), steps: 28, ...props },
  });
}

describe("QualityLadder", () => {
  it("offers the recipe's own ladder as one choice", () => {
    const wrapper = factory();
    expect(wrapper.find("[role='radiogroup']").exists()).toBe(true);
    const rows = wrapper.findAll("[data-test^='quality-row-']");
    expect(rows.map((row) => row.text())).toEqual([
      expect.stringContaining("Draft"),
      expect.stringContaining("Good"),
      expect.stringContaining("Best"),
    ]);
  });

  // "passes", never "steps": the rail's Detail slider counts the same thing.
  it("counts passes beside each row", () => {
    expect(factory().get("[data-test='quality-row-draft']").text()).toContain(
      "8 passes",
    );
  });

  it("marks the row the current pass count sits on", () => {
    const wrapper = factory({ steps: 50 });
    expect(
      wrapper.get("[data-test='quality-row-best']").attributes("aria-checked"),
    ).toBe("true");
    expect(
      wrapper.get("[data-test='quality-row-good']").attributes("aria-checked"),
    ).toBe("false");
  });

  it("marks nothing for a pass count between the rows", () => {
    const wrapper = factory({ steps: 31 });
    for (const row of wrapper.findAll("[data-test^='quality-row-']")) {
      expect(row.attributes("aria-checked")).toBe("false");
    }
  });

  it("asks for the row's pass count when a row is chosen", async () => {
    const wrapper = factory();
    await wrapper.get("[data-test='quality-row-best']").trigger("click");
    expect(wrapper.emitted("select")).toEqual([[50]]);
  });

  /*
   * Timing is the HOST's answer, and the host does not advertise a per-pass
   * estimate yet, so the ladder never invents a duration: a row is its name
   * and its pass count.
   */
  it("names a row and its pass count, and invents no duration", () => {
    const wrapper = factory();
    expect(wrapper.get("[data-test='quality-row-good']").text()).toContain(
      "28 passes",
    );
    expect(wrapper.text()).not.toContain("~");
  });

  it("renders nothing for a recipe whose passes are pinned", () => {
    const wrapper = factory({ presets: [] });
    expect(wrapper.find("[role='radiogroup']").exists()).toBe(false);
    expect(wrapper.findAll("[data-test^='quality-row-']")).toHaveLength(0);
  });

  it("chooses nothing while disabled", async () => {
    const wrapper = factory({ disabled: true });
    const row = wrapper.get("[data-test='quality-row-best']");
    expect(row.attributes("disabled")).toBeDefined();
    await row.trigger("click");
    expect(wrapper.emitted("select")).toBeUndefined();
  });
});
