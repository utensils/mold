import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import { beforeEach, describe, expect, it } from "vitest";
import { newGenerateForm } from "../lib/generateForm";
import type { ModelEntry } from "../lib/api/types";
import MobileSharedParams from "./MobileSharedParams.vue";

describe("MobileSharedParams video duration", () => {
  beforeEach(() => setActivePinia(createPinia()));

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
