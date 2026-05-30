import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import RunningJobCard from "./RunningJobCard.vue";
import type { Job } from "../composables/useGenerateStream";

function makeJob(): Job {
  return {
    id: "job-upscale",
    request: {
      prompt: "a cat",
      model: "flux-dev:fp16",
      width: 512,
      height: 512,
      steps: 4,
      guidance: 3.5,
      upscale_model: "real-esrgan-x4plus:fp16",
    },
    startedAt: 0,
    controller: new AbortController(),
    progress: {
      stage: "Upscaling",
      step: null,
      totalSteps: null,
      weightBytesLoaded: null,
      weightBytesTotal: null,
      queuePosition: null,
      gpu: null,
      elapsedMs: null,
    },
    result: null,
    error: null,
    state: "running",
    chain: null,
    lastProgressAt: Date.now(),
    serverId: "server-job-upscale",
    workStarted: true,
  };
}

describe("RunningJobCard upscaler status", () => {
  it("shows the selected upscaler on a running generation card", () => {
    const wrapper = mount(RunningJobCard, { props: { job: makeJob() } });

    expect(wrapper.text()).toContain("real-esrgan-x4plus:fp16");
    expect(wrapper.text()).toContain("Upscaling");
  });
});
