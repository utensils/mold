import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ActivityStrip from "./ActivityStrip.vue";
import { sequenceToVM } from "@studio/lib/activity";
import type { ActivityJobVM } from "@studio/lib/activity";
import type { ChainJobSummary } from "@studio/lib/api/chainTypes";
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

function makeSequenceVM(
  overrides: Partial<ChainJobSummary> = {},
  host: { hostId: string; hostLabel: string } = {
    hostId: "origin",
    hostLabel: "this server",
  },
): ActivityJobVM {
  const summary: ChainJobSummary = {
    id: "chain-1",
    state: "running",
    model: "ltx-2-19b-distilled:fp8",
    stage_count: 3,
    current_stage: 1,
    created_at_unix_ms: 10,
    updated_at_unix_ms: 10,
    ...overrides,
  };
  const progress = summary.state === "running" ? { step: 4, total: 8 } : null;
  return sequenceToVM(summary, host, progress);
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

  it("shows a sequence row with model, clip count, state, and stage progress", () => {
    const vm = makeSequenceVM();
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [], sequences: [vm] },
    });
    const row = wrapper.get("[data-test='activity-sequence-chain-1']");
    expect(row.text()).toContain("ltx-2-19b-distilled:fp8");
    expect(row.text()).toContain("3 clips");
    expect(row.text()).toContain("running");
    expect(row.text()).toContain("clip 2/3");
    expect(row.text()).toContain("50%");
  });

  it("is visible when only sequence jobs exist and offers the header actions", async () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [], sequences: [makeSequenceVM({ state: "completed" })] },
    });
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(true);
    await wrapper.get("[data-test='activity-clear-inactive']").trigger("click");
    expect(wrapper.emitted("clear-inactive")).toHaveLength(1);
    await wrapper.get("[data-test='activity-cleanup-disk']").trigger("click");
    expect(wrapper.emitted("cleanup-disk")).toHaveLength(1);
  });

  it("emits state-appropriate sequence actions with the VM attached", async () => {
    const running = makeSequenceVM();
    const failed = makeSequenceVM({ id: "chain-2", state: "failed" });
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [], sequences: [running, failed] },
    });
    await wrapper
      .get("[data-test='activity-sequence-chain-1'] [data-action='cancel']")
      .trigger("click");
    expect(wrapper.emitted("sequence-action")?.[0]).toEqual([
      "cancel",
      running,
    ]);

    const failedRow = wrapper.get("[data-test='activity-sequence-chain-2']");
    expect(failedRow.find("[data-action='resume']").exists()).toBe(true);
    expect(failedRow.find("[data-action='edit']").exists()).toBe(true);
    expect(failedRow.find("[data-action='delete']").exists()).toBe(true);
    await failedRow.get("[data-action='edit']").trigger("click");
    expect(wrapper.emitted("sequence-action")?.at(-1)).toEqual([
      "edit",
      failed,
    ]);
  });

  it("orders active sequences ahead of settled ones", () => {
    const done = makeSequenceVM({
      id: "chain-done",
      state: "completed",
      created_at_unix_ms: 999,
    });
    const running = makeSequenceVM({
      id: "chain-live",
      created_at_unix_ms: 1,
    });
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [], sequences: [done, running] },
    });
    const rows = wrapper.findAll("[data-test^='activity-sequence-']");
    expect(rows[0]?.attributes("data-test")).toBe(
      "activity-sequence-chain-live",
    );
  });

  it("badges the host on sequence rows from other machines", () => {
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [],
        sequences: [
          makeSequenceVM({}, { hostId: "plato-7680", hostLabel: "plato" }),
        ],
      },
    });
    expect(
      wrapper.get("[data-test='activity-sequence-chain-1']").text(),
    ).toContain("plato");
  });

  it("keeps failed jobs visible with the server error until dismissed", async () => {
    const failed = makeJob({
      state: "error",
      error: "LTX-2 audio output is unavailable; set enable_audio=false.",
    });
    const wrapper = mount(ActivityStrip, { props: { jobs: [failed] } });

    expect(wrapper.get("[data-test='activity-error-job-1']").text()).toContain(
      "LTX-2 audio output is unavailable",
    );
    await wrapper.get("[data-test='activity-dismiss-job-1']").trigger("click");
    expect(wrapper.emitted("dismiss")?.[0]).toEqual(["job-1"]);
  });
});
