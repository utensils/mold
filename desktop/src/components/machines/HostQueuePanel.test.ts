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
import { useJobsStore, type QueueEntry } from "../../stores/jobs";

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

async function mountPanel(gpuOrdinals: number[], entries: QueueEntry[]) {
  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      { path: "/generate", component: stub },
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

  it("shows the configurable empty line for a host with nothing queued", async () => {
    const { wrapper } = await mountPanel([], []);
    expect(wrapper.get("[data-test='queue-empty']").text()).toBe("Nothing queued");
  });
});
