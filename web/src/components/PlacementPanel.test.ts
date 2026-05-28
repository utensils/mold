import { describe, expect, it } from "vitest";
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
  const sectionToggle = wrapper.find(
    "button[data-test='placement-section-toggle']",
  );
  if (sectionToggle.exists()) await sectionToggle.trigger("click");
  return wrapper;
}

describe("PlacementPanel", () => {
  it("does not render standalone prompt-side placement controls", () => {
    const wrapper = mount(PlacementPanel, {
      props: {
        modelValue: null,
        family: "flux",
        model: "flux-dev:q4",
        gpus: [{ ordinal: 0, name: "RTX 3090" }],
      },
    });
    expect(wrapper.find("[data-test='component-placement']").exists()).toBe(
      false,
    );
    expect(wrapper.find("select[data-test='tier1-select']").exists()).toBe(
      false,
    );
    expect(
      wrapper.find("button[data-test='placement-section-toggle']").exists(),
    ).toBe(false);
    expect(wrapper.text()).not.toContain("Device placement");
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
