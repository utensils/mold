import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import { beforeEach, describe, expect, it } from "vitest";
import { newGenerateForm } from "../lib/generateForm";
import type { ModelEntry } from "../lib/api/types";
import MobileSharedParams from "./MobileSharedParams.vue";

describe("MobileSharedParams video duration", () => {
  beforeEach(() => setActivePinia(createPinia()));

  it("renders distilled guidance fixed but preserves the stored value", async () => {
    const form = reactive({
      ...newGenerateForm(),
      model: "ltx-2.3-22b-distilled:fp8",
      family: "ltx2",
      guidance: 7,
    });
    const wrapper = mount(MobileSharedParams, {
      props: { form, lastSeed: null },
      global: { stubs: { MobileResolutionPicker: true, MobileSeedPicker: true } },
    });
    const guidance = wrapper.get("input[step='0.1']");
    expect(guidance.attributes("disabled")).toBeDefined();
    expect((guidance.element as HTMLInputElement).value).toBe("1");
    expect(form.guidance).toBe(7);
    expect(wrapper.get("[data-test='mobile-fixed-guidance-hint']").text()).toContain(
      "fixes CFG at 1.0",
    );
    expect(wrapper.get("[data-test='mobile-fixed-guidance-hint']").classes()).toContain(
      "mobile-generate-hint",
    );
    expect(wrapper.get("[data-test='mobile-fixed-guidance-hint']").classes()).not.toContain(
      "mobile-generate-validation",
    );
    form.pipeline = "two-stage";
    await wrapper.vm.$nextTick();
    expect(wrapper.get("input[step='0.1']").attributes("disabled")).toBeUndefined();
  });

  it("shows the model-aware seconds slider for one-shot video", async () => {
    const form = reactive({
      ...newGenerateForm(),
      model: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      frames: 97,
      fps: 24,
    });
    const model = {
      name: form.model,
      family: form.family,
      max_runtime_seconds: 20,
      max_frames_absolute: 604,
      frame_step: 8,
    } as ModelEntry;
    const wrapper = mount(MobileSharedParams, {
      props: { form, durationModel: model, lastSeed: null },
      global: {
        stubs: { MobileResolutionPicker: true, MobileSeedPicker: true },
      },
    });

    const slider = wrapper.get("[data-test='mobile-duration'] input[type='range']");
    expect(wrapper.get("[data-test='mobile-duration']").text()).toContain("4.0s");
    expect(
      wrapper
        .findAll("[data-test='mobile-duration'] .ms-slider__mark b")
        .map((mark) => mark.text()),
    ).toEqual(["1×", "2×", "3×", "4×", "5×", "6×"]);
    await slider.setValue("241");
    expect(form.frames).toBe(241);
  });

  it("keeps sequence FPS visible without duplicating the duration control", () => {
    const form = reactive({
      ...newGenerateForm(),
      model: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
    });
    const wrapper = mount(MobileSharedParams, {
      props: { form, lastSeed: null, showFps: true },
      global: {
        stubs: { MobileResolutionPicker: true, MobileSeedPicker: true },
      },
    });

    expect(wrapper.find("[data-test='mobile-duration']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-sequence-fps']").exists()).toBe(true);
  });
});
