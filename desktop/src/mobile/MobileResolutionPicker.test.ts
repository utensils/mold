import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { defineComponent, reactive } from "vue";
import { describe, expect, it } from "vitest";
import MobileResolutionPicker from "./MobileResolutionPicker.vue";

interface PickerState {
  width: number;
  height: number;
  family: string;
}

function mountPicker(
  width: number,
  height: number,
  family = "flux",
  sourceDimensions: { width: number; height: number } | null = null,
) {
  const state = reactive<PickerState>({ width, height, family });
  const Harness = defineComponent({
    components: { MobileResolutionPicker },
    setup: () => ({ state, sourceDimensions }),
    template: `
      <MobileResolutionPicker
        v-model:width="state.width"
        v-model:height="state.height"
        :family="state.family"
        :source-dimensions="sourceDimensions"
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

function tierSegments(wrapper: VueWrapper) {
  return wrapper.get("[data-test='mobile-resolution-tier']").findAll("button");
}

describe("MobileResolutionPicker", () => {
  it("reuses family presets and applies an orientation-aware resolution", async () => {
    const { wrapper, state } = mountPicker(1024, 1024, "flux");

    expect(wrapper.find("[data-test='mobile-resolution-summary']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-resolution-announcement']").text()).toBe(
      "Selected resolution: 1024 by 1024 pixels, 1:1, Square.",
    );
    expect(wrapper.get("[data-orientation='square']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.get("[data-aspect='1:1']").attributes("aria-pressed")).toBe("true");
    const active = wrapper.get("[data-test='mobile-resolution-tier'] [aria-checked='true']");
    expect(active.get(".ms-seg__label").text()).toBe("1 MP");
    expect(active.get(".ms-seg__sub").text()).toBe("High");

    await wrapper.get("[data-orientation='portrait']").trigger("click");
    expect(state).toMatchObject({ width: 768, height: 1024 });
    expect(wrapper.get("[data-test='mobile-resolution-announcement']").text()).toBe(
      "Selected resolution: 768 by 1024 pixels, 3:4, Portrait.",
    );
    expect(wrapper.get("[data-aspect='3:4']").attributes("aria-pressed")).toBe("true");

    await wrapper.get("[data-aspect='9:16']").trigger("click");
    expect(state).toMatchObject({ width: 576, height: 1024 });
  });

  it("offers named resolution tiers for repeated Qwen aspect buckets", async () => {
    const { wrapper, state } = mountPicker(1024, 1024, "qwen-image");
    const segments = tierSegments(wrapper);

    expect(segments).toHaveLength(4);
    expect(segments.map((segment) => segment.get(".ms-seg__label").text())).toEqual([
      "0.3 MP",
      "0.6 MP",
      "1 MP",
      "1.8 MP",
    ]);
    expect(segments.map((segment) => segment.get(".ms-seg__sub").text())).toEqual([
      "Compact",
      "Standard",
      "High",
      "Max",
    ]);
    expect(segments[2]?.attributes("aria-checked")).toBe("true");

    await segments[3]?.trigger("click");
    expect(state).toMatchObject({ width: 1328, height: 1328 });

    await wrapper.get("[data-orientation='portrait']").trigger("click");
    expect(state).toMatchObject({ width: 928, height: 1664 });
    expect(wrapper.get("[data-test='mobile-resolution-announcement']").text()).toBe(
      "Selected resolution: 928 by 1664 pixels, ≈9:16, Portrait.",
    );

    await wrapper.get("[data-aspect='4:7']").trigger("click");
    expect(state).toMatchObject({ width: 768, height: 1344 });
  });

  it("projects the selected tier's exact pixel dimensions under the control", async () => {
    const { wrapper, state } = mountPicker(1024, 1024, "qwen-image");

    expect(wrapper.get("[data-test='mobile-resolution-tier-dims']").text()).toBe("1024 × 1024 px");

    await tierSegments(wrapper)[0]?.trigger("click");
    expect(state).toMatchObject({ width: 512, height: 512 });
    expect(wrapper.get("[data-test='mobile-resolution-tier-dims']").text()).toBe("512 × 512 px");
  });

  it("labels a source-matched custom canvas and restores it after a manual override", async () => {
    const sourceDimensions = { width: 896, height: 1152 };
    const { wrapper, state } = mountPicker(896, 1152, "qwen-image-edit", sourceDimensions);

    expect(wrapper.get("[data-test='mobile-source-resolution-status']").text()).toContain(
      "Matches source · 896×1152",
    );
    expect(wrapper.find("[data-test='mobile-match-source-resolution']").exists()).toBe(false);

    state.width = 1024;
    state.height = 1024;
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-source-resolution-status']").text()).toContain(
      "output is 1024×1024",
    );
    await wrapper.get("[data-test='mobile-match-source-resolution']").trigger("click");
    expect(state).toMatchObject(sourceDimensions);
  });

  it("renders however many tiers the family's aspect bucket provides", () => {
    const single = tierSegments(mountPicker(1024, 1024, "flux").wrapper);
    expect(single).toHaveLength(2);
    expect(single.map((segment) => segment.get(".ms-seg__label").text())).toEqual([
      "0.6 MP",
      "1 MP",
    ]);
    expect(single[1]?.attributes("aria-checked")).toBe("true");

    // LTX-2's 16:9 bucket now reaches upstream's shipped 1080p shape
    // (1920x1088 = 2.1 MP) on top of the existing two tiers.
    const trio = tierSegments(mountPicker(1024, 576, "ltx2").wrapper);
    expect(trio.map((segment) => segment.get(".ms-seg__label").text())).toEqual([
      "0.6 MP",
      "0.9 MP",
      "2.1 MP",
    ]);
    expect(trio.map((segment) => segment.get(".ms-seg__sub").text())).toEqual([
      "Compact",
      "Standard",
      "High",
    ]);
    expect(trio[0]?.attributes("aria-checked")).toBe("true");
  });

  it("shows each aspect choice as a proportionally accurate frame", async () => {
    const { wrapper } = mountPicker(928, 1664, "qwen-image");
    const choices = wrapper.findAll(".mobile-resolution-aspect");

    expect(choices.map((choice) => choice.text())).toEqual(["7:9", "13:19", "4:7", "≈9:16"]);
    const frames = choices.map((choice) =>
      choice.get("[data-test='mobile-resolution-aspect-shape']"),
    );
    expect(frames.every((frame) => frame.attributes("aria-hidden") === "true")).toBe(true);
    expect(
      frames.every(
        (frame) => Number.parseFloat((frame.element as HTMLElement).style.height) === 28,
      ),
    ).toBe(true);

    const widths = frames.map((frame) =>
      Number.parseFloat((frame.element as HTMLElement).style.width),
    );
    expect(widths[0]).toBeGreaterThan(widths[1]!);
    expect(widths[1]).toBeGreaterThan(widths[2]!);
    expect(widths[2]).toBeGreaterThan(widths[3]!);
    expect(choices[3]?.attributes("aria-label")).toBe("Approximately 9 by 16 aspect ratio");
  });

  it("falls back to iPhone-friendly custom fields and snaps values to 16", async () => {
    const { wrapper, state } = mountPicker(1000, 777);

    expect(wrapper.find("[data-test='mobile-resolution-tier']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-resolution-tier-dims']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-resolution-custom']").exists()).toBe(true);
    expect(wrapper.get("[data-test='mobile-resolution-announcement']").text()).toBe(
      "Selected resolution: 1000 by 777 pixels, 1000:777, Landscape.",
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

  it("blocks custom sizes above the server's 1.8 MP ceiling", async () => {
    const { wrapper, state } = mountPicker(2000, 2000);
    const picker = wrapper.getComponent(MobileResolutionPicker);

    expect(wrapper.get("[data-test='mobile-resolution-error']").text()).toContain("1.8 MP");
    expect(picker.emitted("validity-change")?.at(-1)).toEqual([false]);

    state.width = 1328;
    state.height = 1328;
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-resolution-error']").exists()).toBe(false);
    expect(picker.emitted("validity-change")?.at(-1)).toEqual([true]);
  });

  it("exposes every orientation recommended by the shared video contract", async () => {
    const { wrapper, state } = mountPicker(1024, 576, "ltx2");

    expect(wrapper.get("[data-orientation='square']").attributes("disabled")).toBeUndefined();
    expect(wrapper.get("[data-orientation='portrait']").attributes("disabled")).toBeUndefined();
    expect(wrapper.get("[data-orientation='landscape']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.findAll(".mobile-resolution-aspect").map((button) => button.text())).toEqual([
      "22:15",
      "3:2",
      "16:9",
    ]);

    await wrapper.get("[data-orientation='portrait']").trigger("click");
    await flushPromises();
    expect(state).toMatchObject({ width: 576, height: 1024 });
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
