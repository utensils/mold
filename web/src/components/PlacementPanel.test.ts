import { beforeEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import PlacementPanel from "./PlacementPanel.vue";

async function mountPanel(props: {
  family: string;
  placement?: import("../types").DevicePlacement | null;
  model?: string;
  component?: string;
}) {
  const wrapper = mount(PlacementPanel, {
    props: {
      modelValue: props.placement ?? null,
      family: props.family,
      model: props.model ?? "flux-dev:q4",
      component: props.component,
      gpus: [
        { ordinal: 0, name: "RTX 3090" },
        { ordinal: 1, name: "RTX 3090" },
      ],
    },
  });
  // Panel defaults collapsed — expand so the existing assertions about the
  // Tier 1 select / advanced toggle still reach DOM.
  const sectionToggle = wrapper.find(
    "button[data-test='placement-section-toggle']",
  );
  if (sectionToggle.exists()) await sectionToggle.trigger("click");
  return wrapper;
}

describe("PlacementPanel", () => {
  beforeEach(() => {
    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }
  });

  it("renders the Tier 1 select with Auto/CPU/GPU options", async () => {
    const wrapper = await mountPanel({ family: "flux" });
    const opts = wrapper.findAll("select[data-test='tier1-select'] option");
    const labels = opts.map((o) => o.text());
    expect(labels).toContain("Auto");
    expect(labels).toContain("CPU");
    expect(labels.some((l) => l.includes("GPU 0"))).toBe(true);
    expect(labels.some((l) => l.includes("GPU 1"))).toBe(true);
  });

  it("hides Tier 1 select when GPU list is empty", async () => {
    const wrapper = mount(PlacementPanel, {
      props: {
        modelValue: null,
        family: "flux",
        model: "flux-dev:q4",
        gpus: [],
      },
    });
    await wrapper
      .find("button[data-test='placement-section-toggle']")
      .trigger("click");
    expect(wrapper.find("select[data-test='tier1-select']").exists()).toBe(
      false,
    );
  });

  it("enables Advanced disclosure for Tier 2 families", async () => {
    const wrapper = await mountPanel({ family: "flux" });
    const toggle = wrapper.find("button[data-test='advanced-toggle']");
    expect(toggle.exists()).toBe(true);
    expect(toggle.attributes("disabled")).toBeUndefined();
  });

  it("disables Advanced disclosure for Tier 1-only families with a tooltip", async () => {
    const wrapper = await mountPanel({ family: "sdxl" });
    const toggle = wrapper.find("button[data-test='advanced-toggle']");
    expect(toggle.attributes("disabled")).toBeDefined();
    expect(toggle.attributes("title")).toMatch(/not yet available/i);
  });

  it("emits update:modelValue when Tier 1 changes", async () => {
    const wrapper = await mountPanel({ family: "flux" });
    const select = wrapper.find("select[data-test='tier1-select']");
    await select.setValue("cpu");
    const emitted = wrapper.emitted("update:modelValue");
    expect(emitted).toBeTruthy();
    const last = emitted!.at(-1)![0] as
      | import("../types").DevicePlacement
      | null;
    expect(last?.text_encoders).toEqual({ kind: "cpu" });
  });

  it("renders Save-as-default button when placement differs from saved", async () => {
    const wrapper = await mountPanel({
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

  it("defaults collapsed — Tier 1 select is hidden until toggled", () => {
    const wrapper = mount(PlacementPanel, {
      props: {
        modelValue: null,
        family: "flux",
        model: "flux-dev:q4",
        gpus: [{ ordinal: 0, name: "RTX 3090" }],
      },
    });
    expect(wrapper.find("select[data-test='tier1-select']").exists()).toBe(
      false,
    );
    expect(
      wrapper.find("button[data-test='placement-section-toggle']").exists(),
    ).toBe(true);
  });

  it("renders a compact component pin without the standalone section chrome", async () => {
    const wrapper = await mountPanel({ family: "flux", component: "vae" });
    expect(
      wrapper.find("button[data-test='placement-section-toggle']").exists(),
    ).toBe(false);
    const select = wrapper.get("[data-test='component-placement-select']");
    expect(select.findAll("option").map((option) => option.text())).toEqual([
      "Auto",
      "CPU",
      "GPU 0",
      "GPU 1",
    ]);
  });

  it("maps component pins to advanced placement fields", async () => {
    const wrapper = await mountPanel({ family: "flux", component: "clip_g" });
    await wrapper
      .get("[data-test='component-placement-select']")
      .setValue("cpu");

    const emitted = wrapper.emitted("update:modelValue");
    const last = emitted!.at(-1)![0] as import("../types").DevicePlacement;
    expect(last).toEqual({
      text_encoders: { kind: "auto" },
      advanced: {
        transformer: { kind: "auto" },
        vae: { kind: "auto" },
        clip_l: null,
        clip_g: { kind: "cpu" },
        t5: null,
        qwen: null,
      },
    });
  });

  it("maps live component names to the family-specific text encoder field", async () => {
    const wrapper = await mountPanel({
      family: "qwen-image",
      component: "text encoder",
    });
    await wrapper
      .get("[data-test='component-placement-select']")
      .setValue("gpu:1");

    const emitted = wrapper.emitted("update:modelValue");
    const last = emitted!.at(-1)![0] as import("../types").DevicePlacement;
    expect(last.advanced?.qwen).toEqual({ kind: "gpu", ordinal: 1 });
    expect(last.advanced?.t5).toBeNull();
  });
});
