import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { defineComponent, reactive } from "vue";
import { describe, expect, it } from "vitest";
import MobileResolutionPicker from "./MobileResolutionPicker.vue";

interface PickerState {
  width: number;
  height: number;
  family: string;
}

function mountPicker(width: number, height: number, family = "flux") {
  const state = reactive<PickerState>({ width, height, family });
  const Harness = defineComponent({
    components: { MobileResolutionPicker },
    setup: () => ({ state }),
    template: `
      <MobileResolutionPicker
        v-model:width="state.width"
        v-model:height="state.height"
        :family="state.family"
      />
    `,
  });
  return { wrapper: mount(Harness), state };
}

function buttonWithText(wrapper: VueWrapper, text: string) {
  const button = wrapper.findAll("button").find((candidate) => candidate.text() === text);
  if (!button) throw new Error(`Missing ${text} button`);
  return button;
}

describe("MobileResolutionPicker", () => {
  it("reuses family presets and applies an orientation-aware resolution", async () => {
    const { wrapper, state } = mountPicker(1024, 1024, "flux");

    expect(wrapper.get("[data-test='mobile-resolution-summary']").text()).toContain("1024 × 1024");
    expect(wrapper.get("[data-test='mobile-resolution-summary']").text()).toContain("1:1 · Square");
    expect(wrapper.get("[data-orientation='square']").attributes("aria-pressed")).toBe("true");

    await wrapper.get("[data-orientation='portrait']").trigger("click");
    expect(state).toMatchObject({ width: 896, height: 1152 });
    expect(wrapper.get("[data-test='mobile-resolution-summary']").text()).toContain(
      "3:4 · Portrait",
    );
    expect(wrapper.get("[data-aspect='3:4']").attributes("aria-pressed")).toBe("true");

    await wrapper.get("[data-aspect='9:16']").trigger("click");
    expect(state).toMatchObject({ width: 768, height: 1344 });
  });

  it("offers named resolution tiers for repeated Qwen aspect buckets", async () => {
    const { wrapper, state } = mountPicker(1024, 1024, "qwen-image");
    const tier = wrapper.get("[data-test='mobile-resolution-tier']");
    const options = tier.findAll("option");

    expect(options).toHaveLength(4);
    expect(options.map((option) => option.text())).toEqual([
      "Compact · 512 × 512 · 0.3 MP",
      "Standard · 768 × 768 · 0.6 MP",
      "High · 1024 × 1024 · 1.0 MP",
      "Max · 1328 × 1328 · 1.8 MP",
    ]);

    await tier.setValue("1:1 · 1328×1328");
    expect(state).toMatchObject({ width: 1328, height: 1328 });

    await wrapper.get("[data-orientation='portrait']").trigger("click");
    expect(state).toMatchObject({ width: 928, height: 1664 });
    expect(wrapper.get("[data-test='mobile-resolution-summary']").text()).toContain(
      "≈9:16 · Portrait",
    );

    await wrapper.get("[data-aspect='4:7']").trigger("click");
    expect(state).toMatchObject({ width: 768, height: 1344 });
  });

  it("falls back to iPhone-friendly custom fields and snaps values to 16", async () => {
    const { wrapper, state } = mountPicker(1000, 777);

    expect(wrapper.find("[data-test='mobile-resolution-tier']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-resolution-custom']").exists()).toBe(true);
    expect(wrapper.get("[data-test='mobile-resolution-summary']").text()).toContain(
      "1000:777 · Landscape",
    );

    const width = wrapper.get("input[aria-label='Custom width']");
    const height = wrapper.get("input[aria-label='Custom height']");
    expect(width.attributes()).toMatchObject({ inputmode: "numeric", min: "64", step: "16" });
    expect(height.attributes()).toMatchObject({ inputmode: "numeric", min: "64", step: "16" });

    await width.setValue("1001");
    await width.trigger("change");
    await height.setValue("20");
    await height.trigger("change");
    expect(state).toMatchObject({ width: 1008, height: 64 });

    await wrapper.get("[aria-label='Swap width and height']").trigger("click");
    expect(state).toMatchObject({ width: 64, height: 1008 });
  });

  it("lets a preset size reveal manual fields without changing dimensions", async () => {
    const { wrapper, state } = mountPicker(1024, 1024);

    expect(wrapper.find("[data-test='mobile-resolution-custom']").exists()).toBe(false);
    const toggle = wrapper.get("[data-test='mobile-resolution-custom-toggle']");
    expect(toggle.attributes("aria-expanded")).toBe("false");

    await toggle.trigger("click");
    expect(wrapper.find("[data-test='mobile-resolution-custom']").exists()).toBe(true);
    expect(state).toMatchObject({ width: 1024, height: 1024 });
    expect(toggle.text()).toBe("Hide custom size");
  });

  it("disables orientations that the desktop family does not recommend", async () => {
    const { wrapper, state } = mountPicker(1024, 576, "ltx2");

    expect(wrapper.get("[data-orientation='square']").attributes()).toHaveProperty("disabled");
    expect(wrapper.get("[data-orientation='portrait']").attributes()).toHaveProperty("disabled");
    expect(wrapper.get("[data-orientation='landscape']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.findAll(".mobile-resolution-aspect").map((button) => button.text())).toEqual([
      "22:15",
      "3:2",
      "16:9",
    ]);

    await wrapper.get("[data-orientation='portrait']").trigger("click");
    await flushPromises();
    expect(state).toMatchObject({ width: 1024, height: 576 });
  });

  it("exposes controlled width and height update events", async () => {
    const wrapper = mount(MobileResolutionPicker, {
      props: { family: "sdxl", width: 1024, height: 1024 },
    });

    await buttonWithText(wrapper, "Portrait").trigger("click");
    expect(wrapper.emitted("update:width")).toEqual([[896]]);
    expect(wrapper.emitted("update:height")).toEqual([[1152]]);
  });
});
