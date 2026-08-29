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
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useGenerationStore } from "../../stores/generation";
import { useJobsStore, type QueueEntry } from "../../stores/jobs";
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

async function mountPanel(
  gpuOrdinals: number[],
  entries: QueueEntry[],
  plan: QueuePlan | null = null,
) {
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
    caps: { canPause: true, canCancelAll: true, canReorder: true },
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
});

describe("HostQueuePanel", () => {
  it("updates queued row status when the host-wide queue is paused", async () => {
    const { wrapper, jobs } = await mountPanel([], [queued("srv-queued", 0, 0)]);
    jobs.queues.local!.paused = true;
    await flushPromises();

    expect(wrapper.get("[data-test='queue-row']").text()).toContain("PAUSED");
  });

  it("offers Resume for restart-paused work while the global gate is open", async () => {
    const entry = { ...queued("srv-paused", 0, 0), state: "paused" as const };
    const { wrapper, jobs } = await mountPanel([], [entry]);
    const resume = vi.spyOn(jobs, "resume").mockResolvedValue(undefined);

    expect(wrapper.get("[data-test='paused-chip']").text()).toContain("RESTART");
    expect(wrapper.get("[data-test='pause-toggle']").text()).toBe("Resume");
    expect(wrapper.get("[data-test='cancel-entry']").text()).toBe("Cancel");
    await wrapper.get("[data-test='pause-toggle']").trigger("click");
    await flushPromises();
    expect(resume).toHaveBeenCalledWith("local");
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
    expect(wrapper.get("[data-test='pause-toggle']").text()).toBe("Resume");
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

    // ConfirmDialog teleports to the body, so it is not inside the wrapper.
    const accept = document.querySelector<HTMLButtonElement>("[data-test='confirm-accept']");
    accept?.click();
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
});
