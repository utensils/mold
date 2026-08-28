import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiJsonTo: vi.fn().mockResolvedValue({ queue_depth: 0 }),
  apiFetchTo: vi.fn().mockResolvedValue(new Response(null, { status: 200 })),
  currentTarget: () => ({ baseUrl: "http://127.0.0.1:49152", apiKey: null }),
}));
vi.mock("../../lib/api/sse", () => ({ sseStream: vi.fn() }));
vi.mock("../../lib/notify", () => ({ notifyGenerated: vi.fn(), notifyGenerationFailed: vi.fn() }));
vi.mock("../../lib/ipc", () => ({ inTauri: () => false, ipc: {} }));

import QueueColumn from "./QueueColumn.vue";
import { useConnectionStore } from "../../stores/connection";
import { useGenerationStore, type Job } from "../../stores/generation";
import { useJobsStore, type HostQueueCaps, type QueueEntry } from "../../stores/jobs";

function entry(over: Partial<QueueEntry> = {}): QueueEntry {
  return {
    id: "srv-1",
    model: "flux2-klein",
    state: "queued",
    started_at_unix_ms: 1,
    position: 1,
    ...over,
  };
}

function seed(caps: Partial<HostQueueCaps> = {}) {
  const jobs = useJobsStore();
  jobs.queues.local = {
    hostId: "local",
    entries: [entry()],
    paused: null,
    caps: { canPause: true, canCancelAll: true, canReorder: false, ...caps },
    gpuOrdinals: [],
    error: null,
  };
  return jobs;
}

async function mountColumn() {
  setActivePinia(createPinia());
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  const jobs = seed({ canReorder: true });
  const wrapper = mount(QueueColumn);
  await flushPromises();
  return { wrapper, jobs };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("QueueColumn", () => {
  it("mirrors the shared queue surface across hosts", async () => {
    const { wrapper } = await mountColumn();
    const rows = wrapper.findAll("[data-test='queue-surface-row']");
    expect(rows).toHaveLength(1);
    expect(rows[0]!.text()).toContain("flux2-klein");
    expect(rows[0]!.text()).toContain("queued · This device");
  });

  it("updates queued row status when the host-wide queue is paused", async () => {
    const { wrapper, jobs } = await mountColumn();
    jobs.queues.local!.paused = true;
    await flushPromises();

    expect(wrapper.get("[data-test='queue-surface-row']").text()).toContain("paused · This device");
  });

  it("labels a running print as finalizing after denoising completes", async () => {
    setActivePinia(createPinia());
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
    const jobs = useJobsStore();
    jobs.queues.local = {
      hostId: "local",
      entries: [entry({ state: "running", position: 0 })],
      paused: null,
      caps: { canPause: true, canCancelAll: true, canReorder: false },
      gpuOrdinals: [],
      error: null,
    };
    useGenerationStore().jobs = [
      {
        id: "srv-1",
        clientId: 7,
        prompt: "a lighthouse",
        status: "finishing",
        step: 10,
        total: 10,
        error: null,
        submittedAtUnixMs: Date.now(),
        settledAtMs: null,
      } as Job,
    ];
    const wrapper = mount(QueueColumn);
    await flushPromises();

    const text = wrapper.get("[data-test='queue-surface-row']").text();
    expect(text).toContain("finalizing · This device");
    expect(text).not.toContain("developing · This device");
  });

  it("keeps deeper host work reachable through an explicit continuation", async () => {
    const { wrapper, jobs } = await mountColumn();
    jobs.queues.local!.pageLimit = 1;
    jobs.queues.local!.nextCursor = "next";
    const loadMore = vi.spyOn(jobs, "loadMoreHost").mockResolvedValue(undefined);
    await flushPromises();

    await wrapper.get("[data-test='queue-column-load-more']").trigger("click");
    expect(loadMore).toHaveBeenCalledWith("local");
  });

  it("shows a held row as held, not as something still waiting its turn", async () => {
    // Held rows only reach this column because the store stopped filtering
    // them out; without this they would render as "queued · This device",
    // which is the same lie in a new place.
    setActivePinia(createPinia());
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
    const jobs = useJobsStore();
    jobs.queues.local = {
      hostId: "local",
      entries: [entry({ state: "held", held_reason: "dispatch attempts exhausted" })],
      paused: null,
      caps: { canPause: true, canCancelAll: true, canReorder: false },
      gpuOrdinals: [],
      error: null,
    };
    const wrapper = mount(QueueColumn);
    await flushPromises();

    const text = wrapper.get("[data-test='queue-surface-row']").text();
    expect(text).toContain("held");
    expect(text).not.toContain("queued");
  });

  it("cancels a queued row against its owning host", async () => {
    const { wrapper, jobs } = await mountColumn();
    const cancel = vi.spyOn(jobs, "cancelJob").mockResolvedValue(undefined);
    await wrapper.get("[data-test='queue-cancel']").trigger("click");
    await flushPromises();
    expect(cancel).toHaveBeenCalledWith("local", "srv-1");
  });

  it("exposes reorder controls only when the host advertises the capability", async () => {
    const { wrapper, jobs } = await mountColumn();
    const reorder = vi.spyOn(jobs, "reorderQueued").mockResolvedValue(true);
    expect(wrapper.find("[data-test='queue-reorder-up']").exists()).toBe(true);
    await wrapper.get("[data-test='queue-reorder-up']").trigger("click");
    await flushPromises();
    expect(reorder).toHaveBeenCalledWith("local", "srv-1", 0);
  });

  it("reorders within the queued subset, skipping the running job", async () => {
    setActivePinia(createPinia());
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
    const jobs = useJobsStore();
    // [running, A, B]: server positions count the running job, but the reorder
    // PATCH indexes among queued jobs only.
    jobs.queues.local = {
      hostId: "local",
      entries: [
        { id: "run", model: "m", state: "running", started_at_unix_ms: 1, position: 0 },
        { id: "A", model: "m", state: "queued", started_at_unix_ms: 2, position: 1 },
        { id: "B", model: "m", state: "queued", started_at_unix_ms: 3, position: 2 },
      ],
      paused: null,
      caps: { canPause: true, canCancelAll: true, canReorder: true },
      gpuOrdinals: [],
      error: null,
    };
    const reorder = vi.spyOn(jobs, "reorderQueued").mockResolvedValue(true);
    const wrapper = mount(QueueColumn);
    await flushPromises();

    // Only the two queued rows expose reorder buttons: index 0 = A, 1 = B.
    const ups = wrapper.findAll("[data-test='queue-reorder-up']");
    const downs = wrapper.findAll("[data-test='queue-reorder-down']");
    expect(ups).toHaveLength(2);

    await ups[1]!.trigger("click"); // move B up
    expect(reorder).toHaveBeenLastCalledWith("local", "B", 0);

    await downs[0]!.trigger("click"); // move A down
    expect(reorder).toHaveBeenLastCalledWith("local", "A", 1);
  });

  it("hides reorder controls when the capability is absent", async () => {
    setActivePinia(createPinia());
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
    seed({ canReorder: false });
    const wrapper = mount(QueueColumn);
    await flushPromises();
    expect(wrapper.find("[data-test='queue-surface-row']").exists()).toBe(true);
    expect(wrapper.find("[data-test='queue-reorder-up']").exists()).toBe(false);
  });

  it("shows an empty line when nothing is running or queued", async () => {
    setActivePinia(createPinia());
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
    const wrapper = mount(QueueColumn);
    await flushPromises();
    expect(wrapper.get("[data-test='queue-empty']").text()).toContain("Nothing running or queued");
  });
});
