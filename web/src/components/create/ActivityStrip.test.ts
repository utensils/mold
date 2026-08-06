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

const NOW = Date.now();

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
    created_at_unix_ms: NOW - 120_000,
    updated_at_unix_ms: NOW - 60_000,
    ...overrides,
  };
  const progress = summary.state === "running" ? { step: 4, total: 8 } : null;
  return sequenceToVM(summary, host, progress);
}

describe("ActivityStrip", () => {
  it("shows host-attributed shared work without importing it into session jobs", () => {
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [],
        shared: [
          {
            key: "render:download:pull-1",
            id: "pull-1",
            kind: "download",
            phase: "downloading",
            model: "ltx-2",
            hostId: "render",
            hostLabel: "Render box",
            stale: true,
            hostError: "offline",
            created_at_unix_ms: 1,
            updated_at_unix_ms: 2,
            current: 50,
            total: 100,
            can_cancel: false,
          },
        ],
      },
    });
    expect(wrapper.get("[data-test='shared-live-activity']").text()).toContain(
      "Render box",
    );
    expect(wrapper.text()).toContain("Last seen active · offline");
    expect(wrapper.text()).toContain("50%");
  });

  it("is hidden when nothing is in flight", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [makeJob({ state: "done" })] },
    });
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(false);
  });

  it("keeps initiating-client history after the shared terminal snapshot disappears", () => {
    const completed = makeSequenceVM({ state: "completed" });
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [], sequences: [completed], shared: [] },
    });
    expect(wrapper.find("[data-test='shared-live-activity']").exists()).toBe(
      false,
    );
    expect(wrapper.get("[data-test='activity-digest']").text()).toContain(
      "1 settled sequence",
    );
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

  it("opens queued prints and sequences with Space", async () => {
    const queued = makeJob({ id: "job-2", workStarted: false });
    const sequence = makeSequenceVM();
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [queued], sequences: [sequence] },
    });

    await wrapper
      .get("[data-test='activity-queued-job-2']")
      .trigger("keydown", { key: " " });
    expect((wrapper.emitted("open")?.[0]?.[0] as Job).id).toBe(queued.id);

    await wrapper
      .get("[data-test='activity-sequence-chain-1']")
      .trigger("keydown", { key: " " });
    expect(wrapper.emitted("sequence-action")?.[0]).toEqual([
      "watch",
      sequence,
    ]);
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

  it("is visible when only sequence jobs exist, without any maintenance", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [], sequences: [makeSequenceVM({ state: "completed" })] },
    });
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(true);
    // Clear inactive / Clean up disk are destructive, host-scoped, and now
    // live in Library ▸ History ▸ Sequences.
    expect(wrapper.find("[data-test='activity-clear-inactive']").exists()).toBe(
      false,
    );
    expect(wrapper.find("[data-test='activity-cleanup-disk']").exists()).toBe(
      false,
    );
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

  it("orders active sequences ahead of settled-but-wrong ones", () => {
    const failed = makeSequenceVM({
      id: "chain-failed",
      state: "failed",
      error: "boom",
      updated_at_unix_ms: NOW,
    });
    const running = makeSequenceVM({
      id: "chain-live",
      created_at_unix_ms: NOW - 600_000,
    });
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [], sequences: [failed, running] },
    });
    const rows = wrapper.findAll("[data-test^='activity-sequence-']");
    expect(rows[0]?.attributes("data-test")).toBe(
      "activity-sequence-chain-live",
    );
    expect(rows[1]?.attributes("data-test")).toBe(
      "activity-sequence-chain-failed",
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

  it("keeps failed jobs visible without repeating the canvas error", async () => {
    const failed = makeJob({
      state: "error",
      settledAt: NOW - 30_000,
      error: "LTX-2 audio output is unavailable; set enable_audio=false.",
    });
    const wrapper = mount(ActivityStrip, { props: { jobs: [failed] } });

    expect(wrapper.get("[data-test='activity-error-job-1']").text()).toContain(
      "Failed — open Create for details",
    );
    expect(
      wrapper.get("[data-test='activity-error-job-1']").text(),
    ).not.toContain("LTX-2 audio output is unavailable");
    await wrapper
      .get("[data-test='activity-error-job-1']")
      .trigger("keydown", { key: " " });
    expect((wrapper.emitted("open")?.[0]?.[0] as Job).id).toBe(failed.id);
    await wrapper.get("[data-test='activity-dismiss-job-1']").trigger("click");
    expect(wrapper.emitted("dismiss")?.[0]).toEqual(["job-1"]);
  });
});

// "Activity is present tense": settled work resolves to the Library, and the
// strip keeps only a capped, expiring set of rows that still want a decision.
describe("ActivityStrip — present tense", () => {
  it("digests completed sequences instead of listing them", () => {
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [],
        sequences: [
          makeSequenceVM({ id: "a", state: "completed" }),
          makeSequenceVM({ id: "b", state: "completed" }),
        ],
      },
    });
    expect(wrapper.findAll("[data-test^='activity-sequence-']")).toHaveLength(
      0,
    );
    expect(wrapper.get("[data-test='activity-digest']").text()).toBe(
      "2 settled sequences",
    );
  });

  it("keeps a fresh failure with actions but not duplicate detail", async () => {
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [],
        sequences: [
          makeSequenceVM({
            state: "failed",
            error: "stage 2 blew up",
            updated_at_unix_ms: NOW - 60_000,
          }),
        ],
      },
    });
    const row = wrapper.get("[data-test='activity-sequence-chain-1']");
    expect(row.text()).not.toContain("stage 2 blew up");
    expect(row.find("[data-action='resume']").exists()).toBe(true);

    await row
      .get("[data-test='activity-seq-dismiss-chain-1']")
      .trigger("click");
    expect(wrapper.findAll("[data-test^='activity-sequence-']")).toHaveLength(
      0,
    );
    // Dismiss is client-side; the durable job stays on its host.
    expect(wrapper.emitted("sequence-action")).toBeUndefined();
  });

  it("ages a failure out of the strip after five minutes", () => {
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [],
        sequences: [
          makeSequenceVM({
            state: "failed",
            error: "boom",
            updated_at_unix_ms: NOW - 6 * 60_000,
          }),
        ],
      },
    });
    expect(wrapper.findAll("[data-test^='activity-sequence-']")).toHaveLength(
      0,
    );
    expect(wrapper.get("[data-test='activity-digest']").text()).toBe(
      "1 settled sequence",
    );
  });

  it("caps attention rows at two and counts the overflow as failed", () => {
    const failed = (id: string, ago: number) =>
      makeSequenceVM({
        id,
        state: "failed",
        error: "boom",
        updated_at_unix_ms: NOW - ago,
      });
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [],
        sequences: [
          failed("f1", 3_000),
          failed("f2", 2_000),
          failed("f3", 1_000),
        ],
      },
    });
    expect(wrapper.findAll("[data-test^='activity-sequence-']")).toHaveLength(
      2,
    );
    expect(wrapper.get("[data-test='activity-digest']").text()).toContain(
      "1 failed",
    );
  });

  it("asks the page to open History when the digest is clicked", async () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [], sequences: [makeSequenceVM({ state: "completed" })] },
    });
    await wrapper.get("[data-test='activity-digest']").trigger("click");
    expect(wrapper.emitted("show-history")).toHaveLength(1);
  });

  it("hides itself when every job has settled and nothing is left to count", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [makeJob({ state: "done", settledAt: NOW - 1_000 })] },
    });
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(false);
  });
});
