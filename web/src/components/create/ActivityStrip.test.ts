import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ActivityStrip from "./ActivityStrip.vue";
import type { Job } from "../../composables/useGenerateStream";
import type { GenerateRequestWire } from "../../types";

function makeJob(overrides: Partial<Job> = {}): Job {
  const request = {
    prompt: "a cat",
    model: "flux-dev:q4",
    width: 1024,
    height: 1024,
    steps: 28,
    guidance: 3.5,
  } as GenerateRequestWire;
  return {
    id: "job-1",
    request,
    startedAt: 0,
    controller: new AbortController(),
    progress: {
      stage: "Developing",
      step: 14,
      totalSteps: 28,
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
    lastProgressAt: 0,
    workStarted: true,
    serverId: null,
    ...overrides,
  } as Job;
}

describe("ActivityStrip", () => {
  it("is hidden when nothing is in flight", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [makeJob({ state: "done" })] },
    });
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(false);
  });

  it("shows an active job with its prompt and percent", () => {
    const wrapper = mount(ActivityStrip, { props: { jobs: [makeJob()] } });
    expect(wrapper.find("[data-test='activity-running-job-1']").exists()).toBe(
      true,
    );
    expect(wrapper.text()).toContain("a cat");
    expect(wrapper.text()).toContain("50%");
  });

  it("opens a running job on click", async () => {
    const job = makeJob();
    const wrapper = mount(ActivityStrip, { props: { jobs: [job] } });
    await wrapper.get("[data-test='activity-running-job-1']").trigger("click");
    expect((wrapper.emitted("open")?.[0]?.[0] as Job).id).toBe(job.id);
  });

  it("lets the user cancel a running job without opening it", async () => {
    const job = makeJob();
    const wrapper = mount(ActivityStrip, { props: { jobs: [job] } });
    await wrapper.get("[data-test='activity-cancel-job-1']").trigger("click");
    expect(wrapper.emitted("cancel")?.[0]).toEqual(["job-1"]);
    expect(wrapper.emitted("open")).toBeUndefined();
  });

  it("shows queued jobs as cancelable pills", async () => {
    const queued = makeJob({ id: "job-2", workStarted: false });
    const wrapper = mount(ActivityStrip, { props: { jobs: [queued] } });
    expect(wrapper.find("[data-test='activity-queued-job-2']").exists()).toBe(
      true,
    );
    await wrapper.get("[data-test='activity-cancel-job-2']").trigger("click");
    expect(wrapper.emitted("cancel")?.[0]).toEqual(["job-2"]);
  });

  it("falls back to the stage line when no percent is available", () => {
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [
          makeJob({
            progress: {
              stage: "Loading model",
              step: null,
              totalSteps: null,
              weightBytesLoaded: null,
              weightBytesTotal: null,
              queuePosition: null,
              gpu: null,
              elapsedMs: null,
            },
          }),
        ],
      },
    });
    expect(wrapper.text()).toContain("Loading model");
  });
});
