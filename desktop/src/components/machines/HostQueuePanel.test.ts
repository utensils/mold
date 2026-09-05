import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";

vi.mock("../../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiJsonTo: vi.fn().mockResolvedValue({ queue_depth: 0 }),
  apiFetchTo: vi.fn().mockResolvedValue(new Response(null, { status: 200 })),
  currentTarget: () => ({ baseUrl: "http://127.0.0.1:49152", apiKey: null }),
}));
vi.mock("../../lib/api/sse", () => ({ sseStream: vi.fn() }));
vi.mock("../../lib/notify", () => ({ notifyGenerated: vi.fn(), notifyGenerationFailed: vi.fn() }));
vi.mock("../../lib/ipc", () => ({ inTauri: () => false, ipc: {} }));

import HostQueuePanel from "./HostQueuePanel.vue";
import { apiJsonTo } from "../../lib/api/client";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useGenerationStore } from "../../stores/generation";
import { useJobsStore, type QueueEntry } from "../../stores/jobs";
import { useContextMenuStore } from "../../stores/contextMenu";
import { useToastStore } from "../../stores/toasts";
import type { QueuePlan } from "@studio/api/queuePlan";

const stub = { template: "<div />" };
let router: Router;

function queued(id: string, position: number, targetGpu: number): QueueEntry {
  return {
    id,
    model: "flux2-klein",
    state: "queued",
    started_at_unix_ms: 1,
    position,
    target_gpu: targetGpu,
  };
}

function queueMetadata(prompt: string): NonNullable<QueueEntry["metadata"]> {
  return {
    prompt,
    model: "flux2-klein",
    seed: 7,
    steps: 4,
    guidance: 3,
    width: 512,
    height: 512,
  };
}

async function mountPanel(
  gpuOrdinals: number[],
  entries: QueueEntry[],
  plan: QueuePlan | null = null,
) {
  vi.mocked(apiJsonTo).mockImplementation((_target, path) => {
    const match = path.match(/^\/api\/queue\/([^/?]+)$/);
    if (match) {
      const id = decodeURIComponent(match[1]!);
      const entry = entries.find((candidate) => candidate.id === id);
      return entry
        ? Promise.resolve({ job: entry, work_item: null })
        : Promise.reject(new Error(`missing queue fixture ${id}`));
    }
    return Promise.resolve({ queue_depth: 0 });
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const path = new URL(String(input)).pathname;
      const match = path.match(/^\/api\/queue\/([^/]+)$/);
      const id = match ? decodeURIComponent(match[1]!) : "";
      const entry = entries.find((candidate) => candidate.id === id);
      return entry
        ? Response.json({ job: entry, work_item: null })
        : Response.json({ error: `missing queue fixture ${id}` }, { status: 404 });
    }),
  );
  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      { path: "/generate", component: stub },
      { path: "/create", component: stub },
    ],
  });
  router.push("/");
  await router.isReady();
  setActivePinia(createPinia());
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  const hosts = useHostsStore();
  const jobs = useJobsStore();
  jobs.queues.local = {
    hostId: "local",
    entries,
    paused: null,
    caps: { canPause: true, canPauseJob: true, canCancelAll: true, canReorder: true },
    gpuOrdinals,
    plan,
    error: null,
  };
  const host = hosts.all[0]!;
  const wrapper = mount(HostQueuePanel, {
    props: { host },
    global: { plugins: [router], stubs: { DevelopCanvas: stub } },
  });
  await flushPromises();
  return { wrapper, jobs };
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("HostQueuePanel", () => {
  it("updates queued row status when the host-wide queue is paused", async () => {
    const { wrapper, jobs } = await mountPanel([], [queued("srv-queued", 0, 0)]);
    jobs.queues.local!.paused = true;
    await flushPromises();

    expect(wrapper.get("[data-test='queue-row']").text()).toContain("PAUSED");
  });

  it("gives the PAUSED chip one colour, since two in a static class is a coin toss", async () => {
    // Colour utilities are all specificity 0,1,0, so the winner is the order
    // the stylesheet emits them in, not the order of the class attribute. The
    // chip carried both text-fg-dim and text-accent (see styles/kitLayer.test).
    const { wrapper, jobs } = await mountPanel([], [queued("srv-queued", 0, 0)]);
    jobs.queues.local!.paused = true;
    await flushPromises();

    const classes = wrapper.get("[data-test='paused-chip']").classes();
    expect(classes.filter((c) => /^text-(fg|accent|error|warning)/.test(c))).toEqual([
      "text-accent",
    ]);
  });

  it("shows dependency preparation component and progress on a queued row", async () => {
    const entry = queued("preparing", 0, 0);
    const { wrapper } = await mountPanel([], [entry], {
      plan_version: 1,
      state_version: 1,
      optimizer_state: "settled",
      dirty_since_unix_ms: null,
      next_replan_at_unix_ms: null,
      work_items: [
        {
          work_id: entry.id,
          parent_id: entry.id,
          work_kind: "generation",
          priority_class: "user",
          queue_rank: 0,
          bypass_count: 0,
          estimate_confidence: "low",
          blocked_reason: "preparing",
          preparation_progress: {
            component: "Verifying model files",
            bytes_done: 27,
            bytes_total: 100,
          },
        },
      ],
    });

    expect(wrapper.get("[data-test='queue-row']").text()).toContain(
      "PREPARING · VERIFYING MODEL FILES 27%",
    );
  });

  it("shows the host's model-loading stage for a running row", async () => {
    const entry = { ...queued("loading", 0, 0), state: "running" as const };
    const { wrapper } = await mountPanel([], [entry], {
      plan_version: 1,
      state_version: 1,
      optimizer_state: "settled",
      dirty_since_unix_ms: null,
      next_replan_at_unix_ms: null,
      work_items: [
        {
          work_id: entry.id,
          parent_id: entry.id,
          work_kind: "generation",
          priority_class: "user",
          queue_rank: 0,
          bypass_count: 0,
          estimate_confidence: "low",
          activity_phase: "active",
          runtime_phase: "loading",
          runtime_stage: "Loading Flux.2 transformer",
        },
      ],
    });

    expect(wrapper.get("[data-test='queue-row']").text()).toContain("LOADING FLUX.2 TRANSFORMER");
    expect(wrapper.get("[data-test='queue-row']").text()).not.toContain("NEXT UP");
  });

  it("offers per-job Resume while the global gate remains open", async () => {
    const entry = { ...queued("srv-paused", 0, 0), state: "paused" as const };
    const { wrapper, jobs } = await mountPanel([], [entry]);
    const resume = vi.spyOn(jobs, "setJobPaused").mockResolvedValue(undefined);

    expect(wrapper.find("[data-test='paused-chip']").exists()).toBe(false);
    expect(wrapper.get("[data-test='pause-toggle']").text()).toBe("Pause");
    expect(wrapper.get("[data-test='pause-entry']").text()).toBe("Resume");
    expect(wrapper.get("[data-test='cancel-entry']").text()).toBe("Cancel");
    await wrapper.get("[data-test='pause-entry']").trigger("click");
    await flushPromises();
    expect(resume).toHaveBeenCalledWith("local", "srv-paused", false);
  });

  it("does not label other queued work paused when only one row is restart-paused", async () => {
    const pausedEntry = { ...queued("srv-paused", 0, 0), state: "paused" as const };
    const waitingEntry = { ...queued("srv-queued", 1, 0), model: "z-image" };
    const { wrapper } = await mountPanel([], [pausedEntry, waitingEntry]);

    const waitingRow = wrapper
      .findAll("[data-test='queue-row']")
      .find((row) => row.text().includes("z-image"));
    expect(waitingRow?.text()).toContain("QUEUED #1");
    expect(waitingRow?.text()).not.toContain("PAUSED");
    expect(wrapper.get("[data-test='pause-toggle']").text()).toBe("Pause");
  });

  it("pauses only the selected queue row", async () => {
    const { wrapper, jobs } = await mountPanel(
      [],
      [queued("selected", 0, 0), { ...queued("sibling", 1, 0), model: "z-image" }],
    );
    const pause = vi.spyOn(jobs, "setJobPaused").mockResolvedValue(undefined);
    const rows = wrapper.findAll("[data-test='queue-row']");

    await rows[0]!.get("[data-test='pause-entry']").trigger("click");
    await flushPromises();

    expect(pause).toHaveBeenCalledWith("local", "selected", true);
    expect(rows[1]!.text()).not.toContain("PAUSED");
  });

  it("splits a multi-GPU host into per-GPU lanes", async () => {
    const { wrapper } = await mountPanel([0, 1], [queued("srv-0", 1, 0), queued("srv-1", 2, 1)]);
    expect(wrapper.find("[data-test='gpu-lane-0']").exists()).toBe(true);
    expect(wrapper.find("[data-test='gpu-lane-1']").exists()).toBe(true);
    expect(wrapper.findAll("[data-test='queue-row']")).toHaveLength(2);
  });

  it("reassigns a queued row's GPU when dropped on another lane", async () => {
    const { wrapper, jobs } = await mountPanel(
      [0, 1],
      [queued("srv-0", 1, 0), queued("srv-1", 2, 1)],
    );
    const reassign = vi.spyOn(jobs, "reassignGpu").mockResolvedValue(true);
    const rows = wrapper.findAll("[data-test='queue-row']");
    await rows[0]!.trigger("dragstart", { dataTransfer: { setData: vi.fn(), effectAllowed: "" } });
    await wrapper.get("[data-test='gpu-lane-1']").trigger("drop");
    await flushPromises();
    expect(reassign).toHaveBeenCalledWith("local", "srv-0", 1);
  });

  it("opens the info drawer when a row is clicked", async () => {
    const { wrapper } = await mountPanel([], [queued("srv-0", 1, 0)]);
    expect(wrapper.find("[data-test='queue-entry-drawer']").exists()).toBe(false);
    await wrapper.get("[data-test='queue-row']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='queue-entry-drawer']").exists()).toBe(true);
  });

  it("does not reopen a closed drawer when hydration finishes", async () => {
    const { wrapper, jobs } = await mountPanel([], [queued("srv-0", 1, 0)]);
    let resolveDetail!: (entry: QueueEntry) => void;
    vi.spyOn(jobs, "queueJob").mockReturnValue(
      new Promise<QueueEntry>((resolve) => {
        resolveDetail = resolve;
      }),
    );

    await wrapper.get("[data-test='queue-row']").trigger("click");
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await flushPromises();
    expect(wrapper.find("[data-test='queue-entry-drawer']").exists()).toBe(false);

    resolveDetail({ ...queued("srv-0", 1, 0), metadata: queueMetadata("too late") });
    await flushPromises();
    expect(wrapper.find("[data-test='queue-entry-drawer']").exists()).toBe(false);
  });

  it("does not replace a newer drawer selection with stale hydration", async () => {
    const first = queued("srv-0", 1, 0);
    const second = { ...queued("srv-1", 2, 0), model: "z-image" };
    const { wrapper, jobs } = await mountPanel([], [first, second]);
    const resolvers = new Map<string, (entry: QueueEntry) => void>();
    vi.spyOn(jobs, "queueJob").mockImplementation(
      (_hostId, jobId) =>
        new Promise<QueueEntry>((resolve) => {
          resolvers.set(jobId, resolve);
        }),
    );

    const rows = wrapper.findAll("[data-test='queue-row']");
    await rows[0]!.trigger("click");
    await rows[1]!.trigger("click");
    resolvers.get("srv-0")?.({ ...first, metadata: queueMetadata("stale first") });
    await flushPromises();
    expect(wrapper.get("[data-test='queue-entry-drawer']").attributes("aria-label")).toContain(
      "z-image",
    );

    resolvers.get("srv-1")?.({ ...second, metadata: queueMetadata("current second") });
    await flushPromises();
    expect(wrapper.get("[data-test='queue-detail-prompt']").text()).toBe("current second");
  });

  it("shows this app's own submitted request when the durable listing has none", async () => {
    const { wrapper } = await mountPanel([], [queued("srv-0", 1, 0)]);
    const generation = useGenerationStore();
    generation.jobs.push({
      clientId: 7,
      id: "srv-0",
      request: {
        prompt: "a red bicycle",
        model: "flux2-klein",
        width: 1024,
        height: 1024,
      },
    } as never);
    await flushPromises();

    await wrapper.get("[data-test='queue-row']").trigger("click");
    await flushPromises();
    const drawer = wrapper.get("[data-test='queue-entry-drawer']");
    expect(drawer.get("[data-test='queue-detail-prompt']").text()).toBe("a red bicycle");
    expect(drawer.find("[data-test='queue-detail-settings-notice']").exists()).toBe(false);
    expect(drawer.get("[data-test='queue-detail-reuse']").attributes("disabled")).toBeUndefined();
  });

  it("confirms a cancellation from the drawer before touching the host", async () => {
    const { wrapper, jobs } = await mountPanel([], [queued("srv-0", 1, 0)]);
    const cancel = vi.spyOn(jobs, "cancelJob").mockResolvedValue(undefined as never);
    await wrapper.get("[data-test='queue-row']").trigger("click");
    await flushPromises();

    await wrapper.get("[data-test='queue-detail-cancel']").trigger("click");
    await flushPromises();
    expect(cancel).not.toHaveBeenCalled();

    await wrapper.get("[data-test='confirm-accept']").trigger("click");
    await flushPromises();
    expect(cancel).toHaveBeenCalledWith("local", "srv-0");
  });

  it("shows the configurable empty line for a host with nothing queued", async () => {
    const { wrapper } = await mountPanel([], []);
    expect(wrapper.get("[data-test='queue-empty']").text()).toBe("Nothing queued");
  });

  it("offers explicit continuation when the host has another durable page", async () => {
    const { wrapper, jobs } = await mountPanel([], [queued("srv-0", 1, 0)]);
    jobs.queues.local!.pageLimit = 1;
    jobs.queues.local!.nextCursor = "next";
    const loadMore = vi.spyOn(jobs, "loadMoreHost").mockResolvedValue(undefined);
    await flushPromises();

    await wrapper.get("[data-test='queue-load-more']").trigger("click");
    expect(loadMore).toHaveBeenCalledWith("local");
  });

  it("shows scheduler-only work instead of an empty queue", async () => {
    const { wrapper } = await mountPanel([], [], {
      plan_version: 1,
      state_version: 1,
      optimizer_state: "optimized",
      dirty_since_unix_ms: null,
      next_replan_at_unix_ms: null,
      work_items: [
        {
          work_id: "chain-stage-2",
          parent_id: "chain-parent",
          work_kind: "chain_stage",
          chain_stage: 1,
          priority_class: "user",
          queue_rank: 0,
          bypass_count: 0,
          gpu: 2,
          estimate_confidence: "medium",
          activity_phase: "active",
        },
      ],
    });

    expect(wrapper.find("[data-test='queue-empty']").exists()).toBe(false);
    expect(wrapper.findAll("[data-test='planned-queue-row']")).toHaveLength(1);
  });
});

describe("held rows", () => {
  it("labels a held row and explains why, instead of calling it queued", async () => {
    // Held is not waiting — it exhausted its budget and will never start on
    // its own. Rendering it as "QUEUED #2" tells the operator to wait for
    // something that is never coming.
    const { wrapper } = await mountPanel(
      [],
      [
        {
          id: "srv-held",
          model: "flux2-klein",
          state: "held",
          started_at_unix_ms: 1,
          position: 2,
          held_reason: "dispatch attempts exhausted",
        },
      ],
    );

    const text = wrapper.text();
    expect(text).toContain("HELD");
    expect(text).not.toContain("QUEUED #2");
    expect(text).toContain("dispatch attempts exhausted");
  });

  it("offers Cancel on a held row — the one row that cannot clear itself", async () => {
    const { wrapper } = await mountPanel(
      [],
      [
        {
          id: "srv-held",
          model: "flux2-klein",
          state: "held",
          started_at_unix_ms: 1,
          position: 2,
          held_reason: "replay budget exhausted",
        },
      ],
    );

    expect(wrapper.find("[data-test='cancel-entry']").exists()).toBe(true);
  });

  it("hydrates restart-held settings and enables Retry from server authority", async () => {
    const listing: QueueEntry = {
      id: "srv-held",
      model: "flux2-klein",
      state: "held",
      started_at_unix_ms: 1,
      position: 0,
      held_reason: "no enabled GPU",
      retryable: true,
      batch_id: "batch-1",
      client_batch_id: "client-1",
    };
    const { wrapper, jobs } = await mountPanel([], [listing]);
    vi.mocked(fetch).mockResolvedValueOnce(
      Response.json({
        job: {
          ...listing,
          metadata: {
            prompt: "a recovered lighthouse",
            model: "flux2-klein",
            width: 1024,
            height: 1024,
          },
        },
        work_item: null,
      }),
    );
    const retry = vi.spyOn(jobs, "retryJob").mockResolvedValue(undefined);

    await wrapper.get("[data-test='queue-row']").trigger("click");
    await flushPromises();

    const drawer = wrapper.get("[data-test='queue-entry-drawer']");
    expect(drawer.get("[data-test='queue-detail-prompt']").text()).toBe("a recovered lighthouse");
    expect(drawer.get("[data-test='queue-detail-reuse']").attributes("disabled")).toBeUndefined();
    expect(drawer.get("[data-test='queue-detail-retry']").attributes("disabled")).toBeUndefined();

    // The next payload-free queue poll updates state without erasing the
    // explicitly hydrated request or disabling its actions again.
    jobs.queues.local!.entries = [{ ...listing, held_reason: "still waiting for a GPU" }];
    await flushPromises();
    expect(drawer.get("[data-test='queue-detail-prompt']").text()).toBe("a recovered lighthouse");
    expect(drawer.get("[data-test='queue-detail-reuse']").attributes("disabled")).toBeUndefined();

    await drawer.get("[data-test='queue-detail-retry']").trigger("click");
    await flushPromises();
    expect(retry).toHaveBeenCalledWith(
      "local",
      expect.objectContaining({ id: "srv-held", client_batch_id: "client-1" }),
    );
  });

  it("opens a row context menu with queue controls and no disabled retry", async () => {
    const entry: QueueEntry = {
      id: "srv-held",
      model: "flux2-klein",
      state: "held",
      started_at_unix_ms: 1,
      position: 0,
      retryable: true,
      batch_id: "batch-1",
      client_batch_id: "client-1",
    };
    const { wrapper } = await mountPanel([], [entry]);
    await wrapper.get("[data-test='queue-row']").trigger("contextmenu", {
      clientX: 10,
      clientY: 10,
    });
    const items = useContextMenuStore().entries.filter((item) => "label" in item);
    expect(items.map((item) => item.label)).toEqual([
      "Show details",
      "Reuse settings",
      "Retry job",
      "Cancel job",
    ]);
    expect(items.find((item) => item.label === "Retry job")?.disabled).not.toBe(true);
  });

  it("falls back to local settings and retry authority on older hosts", async () => {
    const listing: QueueEntry = {
      id: "srv-held",
      model: "flux2-klein",
      state: "held",
      started_at_unix_ms: 1,
      position: 0,
      retryable: true,
    };
    const { wrapper } = await mountPanel([], [listing]);
    vi.mocked(fetch).mockResolvedValueOnce(Response.json({ error: "not found" }, { status: 404 }));
    const generation = useGenerationStore();
    generation.jobs.push({
      clientId: 42,
      id: "srv-held",
      retryable: true,
      retrying: false,
      request: { prompt: "local recovered prompt", model: "flux2-klein" },
    } as never);
    const retry = vi.spyOn(generation, "retryHeld").mockResolvedValue(undefined);
    await flushPromises();

    await wrapper.get("[data-test='queue-row']").trigger("click");
    await flushPromises();
    const drawer = wrapper.get("[data-test='queue-entry-drawer']");
    expect(drawer.get("[data-test='queue-detail-prompt']").text()).toBe("local recovered prompt");
    expect(drawer.get("[data-test='queue-detail-reuse']").attributes("disabled")).toBeUndefined();
    expect(drawer.get("[data-test='queue-detail-retry']").attributes("disabled")).toBeUndefined();

    vi.mocked(fetch).mockResolvedValueOnce(Response.json({ error: "not found" }, { status: 404 }));
    await drawer.get("[data-test='queue-detail-retry']").trigger("click");
    await flushPromises();
    expect(retry).toHaveBeenCalledWith(42);
  });

  it("reports a context-menu retry failure instead of rejecting globally", async () => {
    const entry: QueueEntry = {
      id: "srv-held",
      model: "flux2-klein",
      state: "held",
      started_at_unix_ms: 1,
      position: 0,
      retryable: true,
      batch_id: "batch-1",
      client_batch_id: "client-1",
    };
    const { wrapper, jobs } = await mountPanel([], [entry]);
    vi.spyOn(jobs, "queueJob").mockRejectedValue(new Error("retry authority changed"));
    await wrapper.get("[data-test='queue-row']").trigger("contextmenu", {
      clientX: 10,
      clientY: 10,
    });
    const retry = useContextMenuStore().entries.find(
      (item) => "label" in item && item.label === "Retry job",
    );
    if (retry && "action" in retry) retry.action?.();
    await flushPromises();

    expect(useToastStore().items.at(-1)).toMatchObject({
      message: "retry authority changed",
      kind: "error",
    });
  });
});
