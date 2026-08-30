import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import ActivityStrip from "./ActivityStrip.vue";
import { useChainJobsStore } from "../../stores/chainJobs";
import { useGenerationStore } from "../../stores/generation";
import { useHostsStore } from "../../stores/hosts";
import { useJobsStore } from "../../stores/jobs";
import { useRunPodStore } from "../../stores/runpod";
import { useLiveActivityStore } from "../../stores/liveActivity";
import { useToastStore } from "../../stores/toasts";
import type { ChainJobSummary } from "@studio/lib/api/chainTypes";
import type { Job } from "../../stores/generation";

const routerPush = vi.fn();
vi.mock("vue-router", () => ({ useRouter: () => ({ push: routerPush }) }));

beforeEach(() => {
  setActivePinia(createPinia());
  routerPush.mockClear();
});
afterEach(() => (document.body.innerHTML = ""));

function baseJob(): Job {
  return {
    clientId: 1,
    prompt: "a lighthouse",
    status: "queued",
    step: 0,
    total: 10,
    error: null,
    submittedAtUnixMs: Date.now(),
    settledAtMs: null,
  } as Job;
}

describe("ActivityStrip", () => {
  it("keeps queued and recovered running work newest-first", () => {
    useGenerationStore().jobs = [
      {
        ...baseJob(),
        prompt: "older queued print",
        submittedAtUnixMs: 1_000,
      },
    ];
    useLiveActivityStore().hosts = {
      render: {
        hostId: "render",
        hostLabel: "Render box",
        target: { baseUrl: "http://render", apiKey: null },
        routeUrl: "http://render",
        instanceId: "render-instance",
        observedAtUnixMs: 3_000,
        stale: false,
        error: null,
        items: [
          {
            id: "newer-running",
            kind: "generation",
            phase: "running",
            model: "newer developing print",
            created_at_unix_ms: 2_000,
            updated_at_unix_ms: 3_000,
            can_cancel: false,
          },
        ],
        unavailableKinds: [],
      },
    };

    const text = mount(ActivityStrip).get("[data-test='activity-list-scroll']").text();
    expect(text.indexOf("newer developing print")).toBeLessThan(text.indexOf("older queued print"));
  });

  it("keeps recovered shared work visible and actionable", async () => {
    useLiveActivityStore().hosts = {
      render: {
        hostId: "render",
        hostLabel: "Render box",
        target: { baseUrl: "http://render", apiKey: null },
        routeUrl: "http://render",
        instanceId: "render-instance",
        observedAtUnixMs: 2,
        stale: true,
        error: "offline",
        items: [
          {
            id: "pull-1",
            kind: "download",
            phase: "downloading",
            model: "ltx-2",
            created_at_unix_ms: 1,
            updated_at_unix_ms: 2,
            current: 50,
            total: 100,
            can_cancel: false,
          },
        ],
        unavailableKinds: [],
      },
    };
    const wrapper = mount(ActivityStrip);
    expect(wrapper.get("[data-test='shared-live-activity']").text()).toContain("Render box");
    expect(wrapper.text()).toContain("Last seen active · offline");
    await wrapper.get("[data-test^='live-activity-select-']").trigger("click");
    expect(routerPush).toHaveBeenCalledWith("/models");
  });

  it("names server-side preparation instead of presenting it as queued", () => {
    useLiveActivityStore().hosts = {
      render: {
        hostId: "render",
        hostLabel: "HAL 9000",
        target: { baseUrl: "http://render", apiKey: null },
        routeUrl: "http://render",
        instanceId: "render-instance",
        observedAtUnixMs: 2,
        stale: false,
        error: null,
        items: [
          {
            id: "print-1",
            kind: "generation",
            phase: "preparing",
            model: "flux2-klein-9b:bf16",
            created_at_unix_ms: 1,
            updated_at_unix_ms: 2,
            current: 27,
            total: 100,
            preparation_progress: {
              component: "Verifying model files",
              bytes_done: 27,
              bytes_total: 100,
            },
            can_cancel: true,
          },
        ],
        unavailableKinds: [],
      },
    };

    const text = mount(ActivityStrip).get("[data-test='shared-live-activity']").text();
    expect(text).toMatch(/HAL 9000 · Preparing · Verifying model files\s+· 27%/);
  });

  it("is hidden when nothing is in flight", () => {
    const wrapper = mount(ActivityStrip);
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(false);
  });

  it("mirrors the running print with its prompt and percent", () => {
    const generation = useGenerationStore();
    generation.jobs = [{ ...baseJob(), status: "denoising", step: 5, total: 10 }];
    const wrapper = mount(ActivityStrip);
    expect(wrapper.get("[data-test='activity-strip']").text()).toContain("a lighthouse");
    expect(wrapper.text()).toContain("50%");
  });

  it("names finalization instead of presenting a bare 100%", () => {
    const generation = useGenerationStore();
    generation.jobs = [{ ...baseJob(), status: "finishing", step: 10, total: 10 }];
    const wrapper = mount(ActivityStrip);
    expect(wrapper.get("[data-test='activity-strip']").text()).toContain("Finalizing");
    expect(wrapper.get("[data-test='activity-strip']").text()).not.toContain("100%");
  });

  it("cancels the running print once and acknowledges the pending request", async () => {
    const generation = useGenerationStore();
    const cancel = vi.spyOn(generation, "cancel").mockResolvedValue(false);
    generation.jobs = [{ ...baseJob(), status: "denoising", cancelling: true, step: 5, total: 10 }];
    const wrapper = mount(ActivityStrip);
    const button = wrapper.get("[data-test='activity-running-cancel']");

    expect(button.attributes("disabled")).toBeDefined();
    expect(button.attributes("aria-label")).toBe("Cancelling a lighthouse");
    await button.trigger("click");
    expect(cancel).not.toHaveBeenCalled();
  });

  it("shows one compact queued print with a working cancel", async () => {
    const generation = useGenerationStore();
    const cancel = vi.spyOn(generation, "cancel").mockResolvedValue(true);
    generation.jobs = [{ ...baseJob(), clientId: 7, status: "queued", prompt: "queued one" }];
    const wrapper = mount(ActivityStrip);
    const pill = wrapper.get("[data-test='activity-queued']");
    expect(pill.text()).toContain("queued one");
    await pill.findAll("button").at(-1)!.trigger("click");
    expect(cancel).toHaveBeenCalledWith(7);
    expect(useToastStore().items.map((item) => item.message)).toContain("Cancelled");
  });

  it("does not claim cancellation when a terminal server event won the race", async () => {
    const generation = useGenerationStore();
    vi.spyOn(generation, "cancel").mockResolvedValue(false);
    generation.jobs = [{ ...baseJob(), clientId: 7, status: "queued" }];
    const wrapper = mount(ActivityStrip);

    await wrapper.get("[data-test='activity-queued']").findAll("button").at(-1)!.trigger("click");

    expect(useToastStore().items).toHaveLength(0);
  });

  it("selects the compact queued print", async () => {
    const generation = useGenerationStore();
    generation.jobs = [{ ...baseJob(), clientId: 7, status: "queued", prompt: "queued one" }];
    const wrapper = mount(ActivityStrip);

    await wrapper.get("[data-test='activity-queued'] button").trigger("click");

    expect(generation.selectedClientId).toBe(7);
  });

  it("keeps one queued print and summarizes the remainder", () => {
    const generation = useGenerationStore();
    generation.jobs = Array.from({ length: 20 }, (_, index) => ({
      ...baseJob(),
      clientId: index + 1,
      prompt: `queued ${index + 1}`,
    }));

    const wrapper = mount(ActivityStrip);

    expect(wrapper.findAll("[data-test='activity-queued']")).toHaveLength(1);
    expect(wrapper.get("[data-test='activity-queued-summary']").text()).toContain("19 queued");
  });

  it("opens the next hidden queued print from the compact summary", async () => {
    const generation = useGenerationStore();
    generation.jobs = [
      { ...baseJob(), clientId: 1, prompt: "oldest", submittedAtUnixMs: 1_000 },
      { ...baseJob(), clientId: 2, prompt: "newest", submittedAtUnixMs: 3_000 },
      { ...baseJob(), clientId: 3, prompt: "next hidden", submittedAtUnixMs: 2_000 },
    ];

    const wrapper = mount(ActivityStrip);

    expect(wrapper.get("[data-test='activity-queued']").text()).toContain("newest");
    await wrapper.get("[data-test='activity-queued-summary']").trigger("click");
    expect(generation.selectedClientId).toBe(3);
  });

  it("collapses queued siblings behind the summary while a print develops", async () => {
    const generation = useGenerationStore();
    generation.jobs = [
      { ...baseJob(), clientId: 1, status: "denoising", prompt: "developing", step: 2 },
      { ...baseJob(), clientId: 2, prompt: "queued older", submittedAtUnixMs: 2_000 },
      { ...baseJob(), clientId: 3, prompt: "queued newest", submittedAtUnixMs: 3_000 },
    ];

    const wrapper = mount(ActivityStrip);

    expect(wrapper.find("[data-test='activity-queued']").exists()).toBe(false);
    expect(wrapper.get("[data-test='activity-running-select']").text()).toContain("developing");
    expect(wrapper.get("[data-test='activity-queued-summary']").text()).toContain("2 queued");

    await wrapper.get("[data-test='activity-queued-summary']").trigger("click");
    expect(generation.selectedClientId).toBe(3);
  });

  it("shows accrued RunPod cost for a job routed to a live pod", () => {
    const generation = useGenerationStore();
    generation.jobs = [
      {
        ...baseJob(),
        status: "denoising",
        hostId: "pod-host",
        step: 1,
      },
    ];
    useHostsStore().extras = [
      {
        id: "pod-host",
        label: "RunPod",
        url: "https://abc123-7680.proxy.runpod.net",
        apiKey: null,
        status: "ready",
        error: null,
        instanceId: null,
      },
    ];
    const runpod = useRunPodStore();
    runpod.loaded = true;
    runpod.overview.pods = [
      {
        id: "abc123",
        desiredStatus: "RUNNING",
        costPerHr: 1.2,
        uptimeSeconds: 3600,
      } as never,
    ];

    const wrapper = mount(ActivityStrip);

    expect(wrapper.get('[data-test="activity-pod-cost"]').text()).toContain("≈$1.20");
  });
});

const NOW = Date.now();

function seqJob(overrides: Partial<ChainJobSummary> = {}): ChainJobSummary {
  return {
    id: "job-1",
    state: "completed",
    model: "ltx-video",
    stage_count: 3,
    current_stage: 2,
    created_at_unix_ms: NOW - 120_000,
    updated_at_unix_ms: NOW - 60_000,
    ...overrides,
  };
}

describe("ActivityStrip — unknown outcomes", () => {
  it("labels a print whose authority was lost as advisory, never failed", () => {
    const generation = useGenerationStore();
    generation.jobs = [
      {
        ...baseJob(),
        status: "error",
        outcomeUnknown: true,
        stage: "Outcome unknown",
        error: "hal9000 was replaced by a new server instance.",
        settledAtMs: Date.now(),
      },
    ];
    const wrapper = mount(ActivityStrip);
    const row = wrapper.get("[data-test='activity-print-attention']");
    expect(row.get(".ms-activity__state").text()).toBe("Outcome unknown");
    expect(row.get(".ms-activity__state").classes()).not.toContain("text-stop");
    expect(row.text()).not.toContain("failed");
    expect(row.get("[data-test='print-dismiss']").attributes("aria-label")).toBe(
      "Dismiss print: a lighthouse",
    );
  });
});

describe("ActivityStrip — sequences", () => {
  it("renders in-flight sequence rows from every host, newest first", () => {
    const chains = useChainJobsStore();
    chains.byHost.local = { jobs: [seqJob({ state: "queued", current_stage: 0 })], error: null };
    chains.byHost["hal9000-7680"] = {
      jobs: [
        seqJob({
          id: "job-2",
          state: "running",
          current_stage: 1,
          created_at_unix_ms: NOW - 600_000,
        }),
      ],
      error: null,
    };
    const wrapper = mount(ActivityStrip);
    const rows = wrapper.findAll("[data-test='activity-sequence']");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.text()).toContain("queued");
    expect(rows[1]!.text()).toContain("running");
    expect(rows[1]!.text()).toContain("2/3");
    expect(rows[0]!.text()).toContain("3 clips");
  });

  it("routes row actions to the owning host", async () => {
    const chains = useChainJobsStore();
    const cancel = vi.spyOn(chains, "cancel").mockResolvedValue();
    const watch = vi.spyOn(chains, "watch").mockReturnValue();
    chains.byHost["hal9000-7680"] = {
      jobs: [seqJob({ state: "running" })],
      error: null,
    };
    const wrapper = mount(ActivityStrip);
    await wrapper.get("[data-test='seq-watch']").trigger("click");
    expect(watch).toHaveBeenCalledWith("hal9000-7680", "job-1");
    await wrapper.get("[data-test='seq-cancel']").trigger("click");
    expect(cancel).toHaveBeenCalledWith("hal9000-7680", "job-1");
  });

  it("surfaces resume for interrupted jobs and emits edit-sequence for Edit", async () => {
    const chains = useChainJobsStore();
    const resume = vi.spyOn(chains, "resume").mockResolvedValue();
    chains.byHost.local = { jobs: [seqJob({ state: "interrupted" })], error: null };
    const wrapper = mount(ActivityStrip);
    await wrapper.get("[data-test='seq-resume']").trigger("click");
    expect(resume).toHaveBeenCalledWith("local", "job-1");
    await wrapper.get("[data-test='seq-edit']").trigger("click");
    expect(wrapper.emitted("edit-sequence")?.[0]).toEqual([{ hostId: "local", jobId: "job-1" }]);
  });

  it("confirms before deleting a sequence job", async () => {
    const chains = useChainJobsStore();
    const remove = vi.spyOn(chains, "remove").mockResolvedValue();
    chains.byHost.local = { jobs: [seqJob({ state: "failed", error: "boom" })], error: null };
    const wrapper = mount(ActivityStrip, { attachTo: document.body });
    await wrapper.get("[data-test='seq-delete']").trigger("click");
    expect(remove).not.toHaveBeenCalled();
    (document.querySelector("[data-test='confirm-accept']") as HTMLButtonElement).click();
    expect(remove).toHaveBeenCalledWith("local", "job-1");
  });

  it("shows the live progress bar for the watched sequence", () => {
    const chains = useChainJobsStore();
    chains.byHost.local = { jobs: [seqJob({ state: "running" })], error: null };
    chains.watching = { hostId: "local", jobId: "job-1" };
    chains.live = {
      detail: null,
      progress: { 1: { step: 4, total: 8 } },
      activeStage: 1,
    };
    const wrapper = mount(ActivityStrip);
    const row = wrapper.get("[data-test='activity-sequence']");
    expect(row.find("[data-test='seq-progress']").exists()).toBe(true);
  });
});

// "Activity is present tense": settled work resolves to the Library, not to a
// growing pile between the canvas and the composer.
describe("ActivityStrip — present tense", () => {
  it("digests completed sequences instead of listing them", () => {
    const chains = useChainJobsStore();
    chains.byHost.local = {
      jobs: [seqJob(), seqJob({ id: "job-2", created_at_unix_ms: NOW - 300_000 })],
      error: null,
    };
    const wrapper = mount(ActivityStrip);
    expect(wrapper.findAll("[data-test='activity-sequence']")).toHaveLength(0);
    expect(wrapper.get("[data-test='activity-digest']").text()).toBe("2 settled sequences");
  });

  it("keeps a fresh failure without repeating its error and allows dismiss", async () => {
    const chains = useChainJobsStore();
    const remove = vi.spyOn(chains, "remove").mockResolvedValue();
    chains.byHost.local = {
      jobs: [
        seqJob({ state: "failed", error: "stage 2 blew up", updated_at_unix_ms: NOW - 60_000 }),
      ],
      error: null,
    };
    const wrapper = mount(ActivityStrip);
    const row = wrapper.get("[data-test='activity-sequence']");
    expect(row.text()).not.toContain("stage 2 blew up");
    expect(row.find("[data-test='seq-resume']").exists()).toBe(true);

    await row.get("[data-test='seq-dismiss']").trigger("click");
    expect(wrapper.findAll("[data-test='activity-sequence']")).toHaveLength(0);
    // Dismiss hides a row; the durable job survives on its host.
    expect(remove).not.toHaveBeenCalled();
  });

  it("ages a failure out of the strip after five minutes", () => {
    const chains = useChainJobsStore();
    chains.byHost.local = {
      jobs: [seqJob({ state: "failed", error: "boom", updated_at_unix_ms: NOW - 6 * 60_000 })],
      error: null,
    };
    const wrapper = mount(ActivityStrip);
    expect(wrapper.findAll("[data-test='activity-sequence']")).toHaveLength(0);
    expect(wrapper.get("[data-test='activity-digest']").text()).toBe("1 settled sequence");
  });

  it("caps attention rows at two and counts the overflow as failed", () => {
    const chains = useChainJobsStore();
    chains.byHost.local = {
      jobs: [
        seqJob({ id: "f1", state: "failed", error: "a", updated_at_unix_ms: NOW - 3_000 }),
        seqJob({ id: "f2", state: "failed", error: "b", updated_at_unix_ms: NOW - 2_000 }),
        seqJob({ id: "f3", state: "failed", error: "c", updated_at_unix_ms: NOW - 1_000 }),
      ],
      error: null,
    };
    const wrapper = mount(ActivityStrip);
    expect(wrapper.findAll("[data-test='activity-sequence']")).toHaveLength(2);
    expect(wrapper.get("[data-test='activity-digest']").text()).toContain("1 failed");
  });

  it("deep-links the digest into Library ▸ History ▸ Sequences", async () => {
    const chains = useChainJobsStore();
    chains.byHost.local = { jobs: [seqJob()], error: null };
    const wrapper = mount(ActivityStrip);
    await wrapper.get("[data-test='activity-digest']").trigger("click");
    expect(routerPush).toHaveBeenCalledWith({
      path: "/library",
      query: { panel: "history", tab: "sequences" },
    });
  });

  it("moves maintenance out of the composer entirely", () => {
    const chains = useChainJobsStore();
    chains.byHost.local = { jobs: [seqJob({ state: "running" })], error: null };
    const wrapper = mount(ActivityStrip);
    expect(wrapper.find("[data-test='activity-clear-inactive']").exists()).toBe(false);
    expect(wrapper.find("[data-test='activity-cleanup']").exists()).toBe(false);
  });

  // The strip is present tense, so its rows only ever say "Watch"; the
  // settled "Open" label is exercised where settled rows live (History ▸
  // Sequences, via ui/components/SequenceJobRow).
  it("labels the watch action Watch for live work", () => {
    const chains = useChainJobsStore();
    chains.byHost.local = { jobs: [seqJob({ state: "running" })], error: null };
    expect(mount(ActivityStrip).get("[data-test='seq-watch']").text()).toBe("Watch");
  });

  it("hides itself when every job has settled and nothing is left to count", () => {
    const generation = useGenerationStore();
    generation.jobs = [{ ...baseJob(), status: "complete", settledAtMs: NOW - 1_000 }];
    const wrapper = mount(ActivityStrip);
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(false);
  });

  it("keeps a failed print visible without repeating its error until dismissed", async () => {
    const generation = useGenerationStore();
    generation.jobs = [
      {
        ...baseJob(),
        status: "error",
        error: "LTX-2 audio output is unavailable.",
        settledAtMs: NOW - 1_000,
      },
    ];
    const wrapper = mount(ActivityStrip);
    const row = wrapper.get("[data-test='activity-print-attention']");
    expect(row.text()).toContain("Open Create for details");
    expect(row.text()).not.toContain("LTX-2 audio output is unavailable.");
    await row.trigger("keydown", { key: " " });
    expect(generation.selectedClientId).toBe(1);
    await row.get("[data-test='print-dismiss']").trigger("click");
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(false);
  });
});

describe("ActivityStrip live queue position", () => {
  // The strip retains the cross-host queue poll while work waits; with no
  // hosts configured a real refresh would prune the seeded snapshot.
  beforeEach(() => {
    const jobs = useJobsStore();
    vi.spyOn(jobs, "refresh").mockResolvedValue(undefined);
  });

  function seedQueue(entries: unknown[], plan: unknown = null) {
    useJobsStore().queues = {
      local: {
        hostId: "local",
        entries: entries as never,
        plan: plan as never,
        paused: null,
        caps: null,
        gpuOrdinals: [],
        error: null,
      },
    };
  }

  it("counts a queued print's live place in line", () => {
    const generation = useGenerationStore();
    generation.jobs = [{ ...baseJob(), id: "srv-2", hostId: "local" }];
    seedQueue([
      { id: "srv-1", model: "m", state: "running", started_at_unix_ms: 1, position: 0 },
      { id: "srv-2", model: "m", state: "queued", started_at_unix_ms: 2, position: 1 },
    ]);
    const wrapper = mount(ActivityStrip);
    expect(wrapper.get("[data-test='activity-queued-position']").text()).toBe("#1 in line");
  });

  it("follows the queue as it drains rather than freezing at the submit slot", async () => {
    const generation = useGenerationStore();
    generation.jobs = [{ ...baseJob(), id: "srv-3", hostId: "local" }];
    seedQueue([{ id: "srv-3", model: "m", state: "queued", started_at_unix_ms: 3, position: 3 }]);
    const wrapper = mount(ActivityStrip);
    expect(wrapper.get("[data-test='activity-queued-position']").text()).toBe("#3 in line");
    seedQueue([{ id: "srv-3", model: "m", state: "queued", started_at_unix_ms: 3, position: 1 }]);
    await wrapper.vm.$nextTick();
    expect(wrapper.get("[data-test='activity-queued-position']").text()).toBe("#1 in line");
  });

  it("says why a job is parked instead of counting its place", () => {
    const generation = useGenerationStore();
    generation.jobs = [{ ...baseJob(), id: "srv-4", hostId: "local" }];
    seedQueue([{ id: "srv-4", model: "m", state: "queued", started_at_unix_ms: 4, position: 2 }], {
      plan_version: 1,
      state_version: 1,
      optimizer_state: "settled",
      dirty_since_unix_ms: null,
      next_replan_at_unix_ms: null,
      work_items: [
        {
          work_id: "w4",
          parent_id: "srv-4",
          work_kind: "generation",
          priority_class: "normal",
          queue_rank: 2,
          bypass_count: 0,
          estimate_confidence: "low",
          blocked_reason: "insufficient_host_ram",
        },
      ],
    });
    const wrapper = mount(ActivityStrip);
    expect(wrapper.get("[data-test='activity-queued-position']").text()).toBe("Waiting for memory");
  });

  it("shows the plain pill against a host that lists nothing", () => {
    const generation = useGenerationStore();
    generation.jobs = [{ ...baseJob(), id: "srv-5", hostId: "local" }];
    const wrapper = mount(ActivityStrip);
    const pill = wrapper.get("[data-test='activity-queued']");
    expect(pill.get("[data-test='activity-queued-position']").text()).toBe("Queued");
    // The pill says it once. The word is the shared label, not chrome around it.
    expect(pill.text().match(/Queued/g)).toHaveLength(1);
  });

  it("counts the line on a busy single-GPU host instead of naming the planner", () => {
    const generation = useGenerationStore();
    generation.jobs = [{ ...baseJob(), id: "srv-6", hostId: "local" }];
    seedQueue([{ id: "srv-6", model: "m", state: "queued", started_at_unix_ms: 6, position: 2 }], {
      plan_version: 1,
      state_version: 1,
      optimizer_state: "settled",
      dirty_since_unix_ms: null,
      next_replan_at_unix_ms: null,
      work_items: [
        {
          work_id: "w6",
          parent_id: "srv-6",
          work_kind: "generation",
          priority_class: "normal",
          queue_rank: 2,
          bypass_count: 0,
          estimate_confidence: "low",
          blocked_reason: "no_idle_device",
        },
      ],
    });
    const wrapper = mount(ActivityStrip);
    expect(wrapper.get("[data-test='activity-queued-position']").text()).toBe("#2 in line");
  });

  it("names the job at the head of the queue", () => {
    const generation = useGenerationStore();
    generation.jobs = [{ ...baseJob(), id: "srv-7", hostId: "local" }];
    seedQueue([{ id: "srv-7", model: "m", state: "queued", started_at_unix_ms: 7, position: 0 }]);
    const wrapper = mount(ActivityStrip);
    expect(wrapper.get("[data-test='activity-queued-position']").text()).toBe("Next up");
  });

  it("retains the shared queue poll only while work is waiting", async () => {
    const jobs = useJobsStore();
    const start = vi.spyOn(jobs, "startPolling").mockImplementation(() => undefined);
    const stop = vi.spyOn(jobs, "stopPolling").mockImplementation(() => undefined);
    const generation = useGenerationStore();
    const wrapper = mount(ActivityStrip);
    expect(start).not.toHaveBeenCalled();
    generation.jobs = [{ ...baseJob(), id: "srv-6", hostId: "local" }];
    await wrapper.vm.$nextTick();
    expect(start).toHaveBeenCalledTimes(1);
    generation.jobs = [{ ...baseJob(), id: "srv-6", hostId: "local", status: "complete" }];
    await wrapper.vm.$nextTick();
    expect(stop).toHaveBeenCalledTimes(1);
  });
});
