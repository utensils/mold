import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ActivityStrip from "./ActivityStrip.vue";
import { buildQueueStatusIndex } from "@studio/lib/queuePosition";
import { ORIGIN_HOST_ID } from "../../lib/hostRegistry";
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

/** A print the stream has already settled as a failure — the only settled
 *  row the strip still holds. */
function makeFailedJob(id: string, settledAgoMs: number): Job {
  return makeJob({
    id,
    state: "error",
    error: "boom",
    settledAt: NOW - settledAgoMs,
  });
}

describe("ActivityStrip", () => {
  it("keeps newer fleet work above an older queued print", () => {
    const olderQueued = makeJob({
      id: "older-queued",
      startedAt: 1_000,
      workStarted: false,
      progress: {
        stage: "Queued",
        step: null,
        totalSteps: null,
        queuePosition: 0,
        gpu: null,
        elapsedMs: null,
      },
    });
    const newerShared = {
      key: "render:generation:newer-running",
      id: "newer-running",
      kind: "generation",
      phase: "running",
      model: "newer developing print",
      hostId: "render",
      hostLabel: "Render box",
      routeUrl: "http://render:7680",
      instanceId: "render-instance",
      stale: false,
      hostError: null,
      created_at_unix_ms: 2_000,
      updated_at_unix_ms: 3_000,
      can_cancel: false,
    };

    const text = mount(ActivityStrip, {
      props: { jobs: [olderQueued], shared: [newerShared] },
    }).text();
    expect(text.indexOf("newer developing print")).toBeLessThan(
      text.indexOf("a cat"),
    );
  });

  it("keeps recovered shared work visible and actionable", async () => {
    const shared = {
      key: "render:download:pull-1",
      id: "pull-1",
      kind: "download" as const,
      phase: "downloading",
      model: "ltx-2",
      hostId: "render",
      hostLabel: "Render box",
      routeUrl: "http://render:7680",
      instanceId: "render-instance",
      stale: true,
      hostError: "offline",
      created_at_unix_ms: 1,
      updated_at_unix_ms: 2,
      current: 50,
      total: 100,
      can_cancel: false,
    };
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [],
        shared: [shared],
      },
    });
    expect(wrapper.get("[data-test='shared-live-activity']").text()).toContain(
      "Render box",
    );
    expect(wrapper.text()).toContain("Last seen active · offline");
    expect(wrapper.text()).toContain("50%");
    await wrapper.get("[data-test^='live-activity-select-']").trigger("click");
    expect(wrapper.emitted("shared-open")?.[0]).toEqual([shared]);
  });

  it("does not label an authority-detached print as failed", () => {
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [
          makeJob({
            state: "error",
            error: "machine replaced",
            detached: true,
            settledAt: Date.now(),
          }),
        ],
      },
    });

    expect(wrapper.text()).toContain(
      "Detached — the original machine still owns the outcome",
    );
    expect(wrapper.text()).not.toContain("Failed — open Create for details");
  });

  it("is hidden when nothing is in flight", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [makeJob({ state: "done" })] },
    });
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(false);
  });

  it("keeps initiating-client history after the shared terminal snapshot disappears", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [makeFailedJob("job-1", 30_000)] },
    });
    expect(wrapper.find("[data-test='shared-live-activity']").exists()).toBe(
      false,
    );
    expect(wrapper.find("[data-test='activity-error-job-1']").exists()).toBe(
      true,
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

  it("names finalization instead of presenting a bare 100%", () => {
    const job = makeJob({
      progress: {
        ...makeJob().progress,
        stage: "Decoding video",
        step: 28,
        totalSteps: 28,
      },
    });
    const wrapper = mount(ActivityStrip, { props: { jobs: [job] } });
    expect(wrapper.text()).toContain("Finalizing");
    expect(wrapper.text()).not.toContain("100%");
  });

  it("prefers the print title over the prompt, and falls back to Untitled print", () => {
    const titled = makeJob({
      request: {
        ...makeJob().request,
        title: "Smurf 04",
      } as GenerateRequestWire,
    });
    const wrapper = mount(ActivityStrip, { props: { jobs: [titled] } });
    expect(wrapper.text()).toContain("Smurf 04");
    expect(wrapper.text()).not.toContain("a cat");

    const blank = makeJob({
      id: "job-2",
      request: { ...makeJob().request, prompt: "   " } as GenerateRequestWire,
    });
    const fallback = mount(ActivityStrip, { props: { jobs: [blank] } });
    expect(fallback.text()).toContain("Untitled print");
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

  it("windows an unlimited queued backlog to one interactive next row plus a summary", () => {
    const jobs = Array.from({ length: 10_000 }, (_, index) =>
      makeJob({
        id: `queued-${index}`,
        startedAt: index,
        workStarted: false,
      }),
    );
    const wrapper = mount(ActivityStrip, { props: { jobs } });

    expect(wrapper.findAll("[data-test^='activity-queued-']")).toHaveLength(2);
    expect(wrapper.findAll(".activity__pill")).toHaveLength(1);
    expect(
      wrapper.get("[data-test='activity-queued-summary']").text(),
    ).toContain("9999 other queued prints");
  });

  it("shows a newer held print above an older queued print", () => {
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [
          makeJob({ id: "older-queued", startedAt: 1, workStarted: false }),
          makeJob({
            id: "newer-held",
            startedAt: 2,
            workStarted: false,
            holdError: "Waiting for memory",
          }),
        ],
      },
    });

    expect(
      wrapper.find("[data-test='activity-queued-newer-held']").exists(),
    ).toBe(true);
    expect(
      wrapper.find("[data-test='activity-queued-older-queued']").exists(),
    ).toBe(false);
    expect(wrapper.text()).toContain("1 other queued print");
  });

  it("reveals the next actionable queued print while an earlier cancel is pending", () => {
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [
          makeJob({
            id: "queued-cancelling",
            startedAt: 0,
            workStarted: false,
            cancelling: true,
            cancelRequested: true,
          }),
          makeJob({ id: "queued-next", startedAt: 1, workStarted: false }),
        ],
      },
    });

    expect(
      wrapper.find("[data-test='activity-queued-queued-next']").exists(),
    ).toBe(true);
    expect(wrapper.text()).toContain("1 other queued print");
  });

  it("opens queued prints with Space", async () => {
    const queued = makeJob({ id: "job-2", workStarted: false });
    const wrapper = mount(ActivityStrip, { props: { jobs: [queued] } });

    await wrapper
      .get("[data-test='activity-queued-job-2']")
      .trigger("keydown", { key: " " });
    expect((wrapper.emitted("open")?.[0]?.[0] as Job).id).toBe(queued.id);
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

  it("renders no sequence rows at all", () => {
    // Authored sequences are retired from the web surface; a chain job the
    // CLI creates reaches this strip only as a read-only fleet row.
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [makeJob()] },
    });
    expect(wrapper.findAll("[data-test^='activity-sequence-']")).toHaveLength(
      0,
    );
    expect(wrapper.find("[data-test='activity-digest']").exists()).toBe(false);
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
  it("keeps a fresh failure with a dismiss control", async () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [makeFailedJob("f1", 60_000)] },
    });
    expect(wrapper.get("[data-test='activity-error-f1']").text()).toContain(
      "Failed — open Create for details",
    );
    await wrapper.get("[data-test='activity-dismiss-f1']").trigger("click");
    expect(wrapper.emitted("dismiss")?.[0]).toEqual(["f1"]);
  });

  it("ages a failure out of the strip after five minutes", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [makeFailedJob("f1", 6 * 60_000)] },
    });
    expect(wrapper.findAll("[data-test^='activity-error-']")).toHaveLength(0);
    // Nothing counts the overflow any more — the digest chip went with the
    // sequence composer.
    expect(wrapper.find("[data-test='activity-digest']").exists()).toBe(false);
  });

  it("caps attention rows at two", () => {
    const wrapper = mount(ActivityStrip, {
      props: {
        jobs: [
          makeFailedJob("f1", 3_000),
          makeFailedJob("f2", 2_000),
          makeFailedJob("f3", 1_000),
        ],
      },
    });
    expect(wrapper.findAll("[data-test^='activity-error-']")).toHaveLength(2);
  });

  it("hides itself when every job has settled and nothing is left to count", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [makeJob({ state: "done", settledAt: NOW - 1_000 })] },
    });
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(false);
  });
});

describe("ActivityStrip live queue position", () => {
  const queued = (serverId: string) =>
    makeJob({
      id: `client-${serverId}`,
      serverId,
      workStarted: false,
      state: "running",
    });

  const index = buildQueueStatusIndex([
    {
      hostId: ORIGIN_HOST_ID,
      entries: [
        {
          id: "srv-1",
          model: "m",
          state: "running",
          started_at_unix_ms: 1,
          position: 0,
        },
        {
          id: "srv-2",
          model: "m",
          state: "queued",
          started_at_unix_ms: 2,
          position: 2,
        },
        {
          id: "srv-3",
          model: "m",
          state: "queued",
          started_at_unix_ms: 3,
          position: 3,
        },
      ],
      plan: {
        plan_version: 1,
        state_version: 1,
        optimizer_state: "settled",
        dirty_since_unix_ms: null,
        next_replan_at_unix_ms: null,
        work_items: [
          {
            work_id: "w3",
            parent_id: "srv-3",
            work_kind: "generation",
            priority_class: "normal",
            queue_rank: 3,
            bypass_count: 0,
            estimate_confidence: "low",
            blocked_reason: "insufficient_host_ram",
          },
          {
            work_id: "w2",
            parent_id: "srv-2",
            work_kind: "generation",
            priority_class: "normal",
            queue_rank: 2,
            bypass_count: 0,
            estimate_confidence: "low",
            // Every GPU busy: ordinary serialization, not a fault.
            blocked_reason: "no_idle_device",
          },
        ],
      },
    },
  ]);

  it("counts a queued print's live place in line", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [queued("srv-2")], queueStatus: index },
    });
    // `no_idle_device` on this row is a busy host, not a stall: the row keeps
    // counting rather than printing the scheduler's own string.
    expect(
      wrapper.get("[data-test='activity-queue-position-client-srv-2']").text(),
    ).toBe("#2 in line");
  });

  it("says why a parked job is waiting instead of counting its place", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [queued("srv-3")], queueStatus: index },
    });
    expect(
      wrapper.get("[data-test='activity-queue-position-client-srv-3']").text(),
    ).toBe("Waiting for memory");
  });

  it("keeps the plain pill against a server that lists nothing", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [queued("srv-9")] },
    });
    expect(
      wrapper.get("[data-test='activity-queue-position-client-srv-9']").text(),
    ).toBe("Queued");
    expect(wrapper.get("[data-test='activity-strip']").text()).toContain(
      "a cat",
    );
  });

  it("names the job at the head of the queue", () => {
    const wrapper = mount(ActivityStrip, {
      props: { jobs: [queued("srv-1")], queueStatus: index },
    });
    expect(
      wrapper.get("[data-test='activity-queue-position-client-srv-1']").text(),
    ).toBe("Next up");
  });
});
