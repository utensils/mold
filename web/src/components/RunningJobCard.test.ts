import { mount, type VueWrapper } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import RunningJobCard from "./RunningJobCard.vue";
import { STALE_THRESHOLD_MS, type Job } from "../composables/useGenerateStream";
import type { QueueEntry } from "../types";

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "job-1",
    request: {
      prompt: "a cat",
      model: "flux-dev:fp16",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
    },
    startedAt: 0,
    controller: new AbortController(),
    progress: {
      stage: "Denoising",
      step: 1,
      totalSteps: 20,
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
    serverId: "server-job-1",
    workStarted: true,
    ...overrides,
  } as Job;
}

describe("RunningJobCard stale stream warning", () => {
  const wrappers: VueWrapper[] = [];

  afterEach(() => {
    for (const wrapper of wrappers.splice(0)) wrapper.unmount();
    vi.useRealTimers();
  });

  function mountAtStaleAge(job: Job): VueWrapper {
    vi.useFakeTimers();
    vi.setSystemTime(STALE_THRESHOLD_MS + 1);
    const wrapper = mount(RunningJobCard, { props: { job } });
    wrappers.push(wrapper);
    return wrapper;
  }

  it("does not warn for queued jobs even when the queue has been quiet past the threshold", () => {
    const wrapper = mountAtStaleAge(
      makeJob({
        progress: {
          ...makeJob().progress,
          stage: "Queued (position 3)",
          queuePosition: 3,
          step: null,
          totalSteps: null,
        },
        workStarted: false,
      } as Partial<Job>),
    );

    expect(wrapper.text()).toContain("Queued (position 3)");
    expect(wrapper.text()).not.toContain("stream may have dropped");
  });

  it("warns once real work has started and progress goes quiet", () => {
    const wrapper = mountAtStaleAge(makeJob());

    expect(wrapper.text()).toContain("stream may have dropped");
  });
});

describe("RunningJobCard lane controls", () => {
  it("does not render a per-card lane dropdown for queued queue entries", () => {
    const queueEntry: QueueEntry = {
      id: "server-job-1",
      model: "flux-dev:fp16",
      state: "queued",
      started_at_unix_ms: 0,
      position: 1,
      target_gpu: null,
    };

    const wrapper = mount(RunningJobCard, {
      props: {
        job: makeJob({ workStarted: false }),
        queueEntry,
      },
    });

    expect(wrapper.find('[data-test="job-lane-select"]').exists()).toBe(false);
    expect(wrapper.emitted("lane-change")).toBeUndefined();
  });
});
