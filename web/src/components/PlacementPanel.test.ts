import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import PlacementPanel from "./PlacementPanel.vue";

function mountPanel(props: {
  family: string;
  placement?: import("../types").DevicePlacement | null;
  model?: string;
}) {
  return mount(PlacementPanel, {
    props: {
      modelValue: props.placement ?? null,
      family: props.family,
      model: props.model ?? "flux-dev:q4",
      gpus: [
        { ordinal: 0, name: "RTX 3090" },
        { ordinal: 1, name: "RTX 3090" },
      ],
    },
  });
}

describe("PlacementPanel", () => {
  it("renders the Tier 1 select with Auto/CPU/GPU options", () => {
    const wrapper = mountPanel({ family: "flux" });
    const opts = wrapper.findAll("select[data-test='tier1-select'] option");
    const labels = opts.map((o) => o.text());
    expect(labels).toContain("Auto");
    expect(labels).toContain("CPU");
    expect(labels.some((l) => l.includes("GPU 0"))).toBe(true);
    expect(labels.some((l) => l.includes("GPU 1"))).toBe(true);
  });

  it("hides Tier 1 select when GPU list is empty", () => {
    const wrapper = mount(PlacementPanel, {
      props: {
        modelValue: null,
        family: "flux",
        model: "flux-dev:q4",
        gpus: [],
      },
    });
    expect(wrapper.find("select[data-test='tier1-select']").exists()).toBe(
      false,
    );
  });

  it("enables Advanced disclosure for Tier 2 families", () => {
    const wrapper = mountPanel({ family: "flux" });
    const toggle = wrapper.find("button[data-test='advanced-toggle']");
    expect(toggle.exists()).toBe(true);
    expect(toggle.attributes("disabled")).toBeUndefined();
  });

  it("disables Advanced disclosure for Tier 1-only families with a tooltip", () => {
    const wrapper = mountPanel({ family: "sdxl" });
    const toggle = wrapper.find("button[data-test='advanced-toggle']");
    expect(toggle.attributes("disabled")).toBeDefined();
    expect(toggle.attributes("title")).toMatch(/not yet available/i);
  });

  it("emits update:modelValue when Tier 1 changes", async () => {
    const wrapper = mountPanel({ family: "flux" });
    const select = wrapper.find("select[data-test='tier1-select']");
    await select.setValue("cpu");
    const emitted = wrapper.emitted("update:modelValue");
    expect(emitted).toBeTruthy();
    const last = emitted!.at(-1)![0] as
      | import("../types").DevicePlacement
      | null;
    expect(last?.text_encoders).toEqual({ kind: "cpu" });
  });

  it("renders Save-as-default button when placement differs from saved", () => {
    const wrapper = mountPanel({
      family: "flux",
      placement: {
        text_encoders: { kind: "cpu" },
        advanced: null,
      },
    });
    expect(wrapper.find("button[data-test='save-default']").exists()).toBe(
      true,
    );
  });
});
