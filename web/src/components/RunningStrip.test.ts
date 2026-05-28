import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import RunningStrip from "./RunningStrip.vue";
import type { Job } from "../composables/useGenerateStream";
import type { GenerateRequestWire, QueueEntry } from "../types";

function makeJob(overrides: Partial<Job> = {}): Job {
  const id = overrides.id ?? "client-1";
  return {
    id,
    request: {
      prompt: "a cat",
      model: "flux-dev:fp16",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
    } as GenerateRequestWire,
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
    serverId: `srv-${id}`,
    workStarted: true,
    ...overrides,
  } as Job;
}

function entry(overrides: Partial<QueueEntry> = {}): QueueEntry {
  return {
    id: "srv-1",
    model: "flux-dev:fp16",
    state: "queued",
    started_at_unix_ms: 0,
    position: 0,
    ...overrides,
  };
}

describe("RunningStrip queue lanes", () => {
  it("keeps running cards to the left of queued cards in each lane", () => {
    const wrapper = mount(RunningStrip, {
      props: {
        jobs: [
          makeJob({
            id: "queued-client",
            serverId: "srv-queued",
            workStarted: false,
            progress: {
              ...makeJob().progress,
              stage: "Queued (position 4)",
              queuePosition: 4,
              gpu: null,
            },
          }),
          makeJob({
            id: "running-client",
            serverId: "srv-running",
            progress: { ...makeJob().progress, gpu: 0 },
          }),
        ],
        queueEntries: [
          entry({ id: "srv-queued", state: "queued", position: 4 }),
          entry({ id: "srv-running", state: "running", position: 0, gpu: 0 }),
        ],
        gpus: [{ ordinal: 0, state: "busy" }],
      },
    });

    const labels = wrapper
      .findAll('[data-test="running-card"]')
      .map((n) => n.text());
    expect(labels[0]).toContain("GPU 0");
    expect(labels[0]).toContain("Denoising");
    expect(labels[1]).toContain("Queued (position 4)");
  });

  it("places unassigned queued jobs into visible GPU lanes without rendering Auto", () => {
    const wrapper = mount(RunningStrip, {
      props: {
        jobs: [makeJob({ id: "running", serverId: "srv-running" })],
        queueEntries: [
          entry({
            id: "srv-auto",
            model: "sdxl:q8",
            state: "queued",
            position: 2,
          }),
          entry({
            id: "srv-targeted",
            model: "flux-dev:q4",
            state: "queued",
            position: 1,
            target_gpu: 1,
          }),
          entry({ id: "srv-running", state: "running", position: 0, gpu: 0 }),
        ],
        gpus: [
          { ordinal: 0, state: "busy" },
          { ordinal: 1, state: "idle" },
        ],
      },
    });

    expect(wrapper.find('[data-test="queue-lane-auto"]').exists()).toBe(false);
    expect(wrapper.get('[data-test="queue-lane-gpu-0"]').text()).toContain(
      "GPU 0",
    );
    expect(wrapper.get('[data-test="queue-lane-gpu-0"]').text()).toContain(
      "sdxl:q8",
    );
    expect(wrapper.get('[data-test="queue-lane-gpu-1"]').text()).toContain(
      "flux-dev:q4",
    );
  });

  it("opens another GPU lane for unassigned jobs only after the lane threshold is reached", async () => {
    const wrapper = mount(RunningStrip, {
      props: {
        jobs: [],
        queueEntries: [
          entry({
            id: "srv-a",
            model: "model-a",
            state: "queued",
            position: 0,
          }),
          entry({
            id: "srv-b",
            model: "model-b",
            state: "queued",
            position: 1,
          }),
          entry({
            id: "srv-c",
            model: "model-c",
            state: "queued",
            position: 2,
          }),
        ],
        gpus: [
          { ordinal: 0, state: "idle" },
          { ordinal: 1, state: "idle" },
        ],
      },
    });

    expect(wrapper.get('[data-test="queue-lane-gpu-0"]').text()).toContain(
      "model-a",
    );
    expect(wrapper.get('[data-test="queue-lane-gpu-0"]').text()).toContain(
      "model-b",
    );
    expect(wrapper.get('[data-test="queue-lane-gpu-1"]').text()).toContain(
      "model-c",
    );

    await wrapper.get('[data-test="queue-lane-threshold"]').setValue("1");

    expect(wrapper.get('[data-test="queue-lane-gpu-0"]').text()).toContain(
      "model-a",
    );
    expect(wrapper.get('[data-test="queue-lane-gpu-1"]').text()).toContain(
      "model-b",
    );
    expect(wrapper.get('[data-test="queue-lane-gpu-1"]').text()).toContain(
      "model-c",
    );
  });

  it("emits lane changes when queued cards are dragged between GPU lanes", async () => {
    const wrapper = mount(RunningStrip, {
      props: {
        jobs: [],
        queueEntries: [
          entry({ id: "srv-queued", state: "queued", position: 1 }),
          entry({ id: "srv-running", state: "running", position: 0, gpu: 0 }),
        ],
        gpus: [
          { ordinal: 0, state: "busy" },
          { ordinal: 1, state: "idle" },
        ],
      },
    });

    const queuedCard = wrapper.find('[data-queue-id="srv-queued"]');
    const runningCard = wrapper.find('[data-queue-id="srv-running"]');

    expect(queuedCard.attributes("draggable")).toBe("true");
    expect(runningCard.attributes("draggable")).toBe("false");
    expect(wrapper.find('[data-test="job-lane-select"]').exists()).toBe(false);

    await queuedCard.trigger("dragstart");
    await wrapper.get('[data-test="queue-lane-gpu-1"]').trigger("drop");

    expect(wrapper.emitted("lane-change")).toEqual([["srv-queued", 1]]);
  });

  it("renders queued server rows even when the local stream job is missing", () => {
    const wrapper = mount(RunningStrip, {
      props: {
        jobs: [],
        queueEntries: [
          entry({ id: "srv-server-only", model: "qwen-image:q8", position: 7 }),
        ],
        gpus: [],
      },
    });

    expect(wrapper.text()).toContain("qwen-image:q8");
    expect(wrapper.text()).toContain("Queued (position 7)");
  });
});
