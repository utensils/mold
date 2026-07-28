import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import ActivityStrip from "./ActivityStrip.vue";
import { useChainJobsStore } from "../../stores/chainJobs";
import { useGenerationStore } from "../../stores/generation";
import { useHostsStore } from "../../stores/hosts";
import { useRunPodStore } from "../../stores/runpod";
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

  it("lists queued siblings with a working cancel", async () => {
    const generation = useGenerationStore();
    const cancel = vi.spyOn(generation, "cancel").mockResolvedValue(undefined as never);
    generation.jobs = [{ ...baseJob(), clientId: 7, status: "queued", prompt: "queued one" }];
    const wrapper = mount(ActivityStrip);
    const pill = wrapper.get("[data-test='activity-queued']");
    expect(pill.text()).toContain("queued one");
    await pill.get("button").trigger("click");
    expect(cancel).toHaveBeenCalledWith(7);
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

describe("ActivityStrip — sequences", () => {
  it("renders in-flight sequence rows from every host, running work first", () => {
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
    // The shared merge puts active work first even when it is older.
    expect(rows[0]!.text()).toContain("running");
    expect(rows[0]!.text()).toContain("2/3");
    expect(rows[0]!.text()).toContain("3 clips");
    expect(rows[1]!.text()).toContain("queued");
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

  it("keeps a fresh failure with its error and a non-destructive dismiss", async () => {
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
    expect(row.text()).toContain("stage 2 blew up");
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

  it("keeps a failed print visible with its error until dismissed", async () => {
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
    expect(row.text()).toContain("LTX-2 audio output is unavailable.");
    await row.get("[data-test='print-dismiss']").trigger("click");
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(false);
  });
});
