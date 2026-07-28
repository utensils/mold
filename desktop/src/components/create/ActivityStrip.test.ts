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

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => (document.body.innerHTML = ""));

function baseJob(): Job {
  return {
    clientId: 1,
    prompt: "a lighthouse",
    status: "queued",
    step: 0,
    total: 10,
    error: null,
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

function seqJob(overrides: Partial<ChainJobSummary> = {}): ChainJobSummary {
  return {
    id: "job-1",
    state: "completed",
    model: "ltx-video",
    stage_count: 3,
    current_stage: 2,
    created_at_unix_ms: 10,
    updated_at_unix_ms: 20,
    ...overrides,
  };
}

describe("ActivityStrip — sequences", () => {
  it("renders sequence rows from every host, running work first", () => {
    const chains = useChainJobsStore();
    chains.byHost.local = { jobs: [seqJob()], error: null };
    chains.byHost["hal9000-7680"] = {
      jobs: [seqJob({ id: "job-2", state: "running", current_stage: 1, created_at_unix_ms: 1 })],
      error: null,
    };
    const wrapper = mount(ActivityStrip);
    const rows = wrapper.findAll("[data-test='activity-sequence']");
    expect(rows).toHaveLength(2);
    // The shared merge puts active work first even when it is older.
    expect(rows[0]!.text()).toContain("running");
    expect(rows[0]!.text()).toContain("2/3");
    expect(rows[0]!.text()).toContain("3 clips");
    expect(rows[1]!.text()).toContain("completed");
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
    chains.byHost.local = { jobs: [seqJob()], error: null };
    const wrapper = mount(ActivityStrip, { attachTo: document.body });
    await wrapper.get("[data-test='seq-delete']").trigger("click");
    expect(remove).not.toHaveBeenCalled();
    (document.querySelector("[data-test='confirm-accept']") as HTMLButtonElement).click();
    expect(remove).toHaveBeenCalledWith("local", "job-1");
  });

  it("strip-level Clear inactive clears sequences and prunes finished prints", async () => {
    const chains = useChainJobsStore();
    const clear = vi.spyOn(chains, "clearInactive").mockResolvedValue({ cleared: 1, failed: 0 });
    const generation = useGenerationStore();
    const prune = vi.spyOn(generation, "prune");
    chains.byHost.local = { jobs: [seqJob()], error: null };
    const wrapper = mount(ActivityStrip);
    await wrapper.get("[data-test='activity-clear-inactive']").trigger("click");
    expect(clear).toHaveBeenCalledWith();
    expect(prune).toHaveBeenCalledWith(0);
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
