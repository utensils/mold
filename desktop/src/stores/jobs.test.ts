import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

const apiJsonTo = vi.fn();
const apiFetchTo = vi.fn().mockResolvedValue(new Response(null, { status: 200 }));
const listDevices = vi.fn();
vi.mock("../lib/api/client", () => ({
  apiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
  apiFetchTo: (...a: unknown[]) => apiFetchTo(...a),
}));
vi.mock("@studio/api/devices", () => ({
  listDevices: (...a: unknown[]) => listDevices(...a),
}));

import { useConnectionStore } from "./connection";
import { useHostsStore } from "./hosts";
import { enrichQueueEntries, useJobsStore } from "./jobs";
import { useToastStore } from "./toasts";
import type { Job } from "./generation";
import type { DeviceInfo } from "@studio/api/devices";

function device(ordinal: number, overrides: Partial<DeviceInfo> = {}): DeviceInfo {
  return {
    id: `cuda:${ordinal}`,
    backend: "cuda",
    ordinal,
    device_kind: "full_gpu",
    nvml_uuid: `GPU-${ordinal}`,
    physical_uuid: `GPU-${ordinal}`,
    mig_uuid: null,
    mig_parent_uuid: null,
    mig_profile: null,
    name: `GPU ${ordinal}`,
    pci_bus_id: null,
    compute_capability: "8.6",
    memory: {
      total_bytes: 24 * 1024 ** 3,
      used_bytes: 0,
      mold_used_bytes: 0,
      other_used_bytes: 0,
    },
    telemetry: {
      utilization_percent: 0,
      temperature_c: 30,
      power_w: 20,
    },
    desired_enabled: true,
    admin_state: "enabled",
    health: "healthy",
    activity: "idle",
    schedulable: true,
    unschedulable_reason: null,
    loaded_models: [],
    active_work_id: null,
    planned_work_ids: [],
    ...overrides,
  };
}

function seedHosts() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
  conn.status = "ready";
  const hosts = useHostsStore();
  hosts.extras.push({
    id: "hal9000-7680",
    label: "hal9000",
    url: "http://hal9000:7680",
    apiKey: "hk",
    status: "ready",
    error: null,
    instanceId: null,
  });
  return hosts;
}

/** Route API mocks by path so one implementation serves every host. */
function installApi({
  paused = false,
  gpus,
}: { paused?: boolean; gpus?: { ordinal: number }[] } = {}) {
  apiJsonTo.mockImplementation((_target: unknown, path: string) => {
    if (path === "/api/queue") {
      return Promise.resolve({
        entries: [
          {
            id: "srv-1",
            model: "flux2-klein",
            state: "running",
            started_at_unix_ms: 1,
            position: 0,
            gpu: 0,
          },
          { id: "srv-2", model: "sdxl:q8", state: "queued", started_at_unix_ms: 2, position: 1 },
        ],
      });
    }
    if (path === "/api/status")
      return Promise.resolve({ version: "0.17.0", queue_paused: paused, gpus });
    if (path === "/api/capabilities") {
      return Promise.resolve({
        queue: { can_pause: true, can_cancel_all: true, can_reorder: true },
      });
    }
    return Promise.reject(new Error(`unexpected path ${path}`));
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  apiFetchTo.mockResolvedValue(new Response(null, { status: 200 }));
  listDevices.mockRejectedValue(new Error("legacy server"));
});

describe("jobs store", () => {
  it("refresh() snapshots every ready host's queue, pause state, and capabilities", async () => {
    seedHosts();
    installApi({ paused: true });
    const jobs = useJobsStore();
    await jobs.refresh();
    expect(Object.keys(jobs.queues).sort()).toEqual(["hal9000-7680", "local"]);
    const q = jobs.queues["local"]!;
    expect(q.entries).toHaveLength(2);
    expect(q.paused).toBe(true);
    expect(q.caps).toEqual({ canPause: true, canCancelAll: true, canReorder: true });
  });

  it("refreshHost() snapshots only the given host", async () => {
    const hosts = seedHosts();
    installApi();
    const jobs = useJobsStore();
    await jobs.refreshHost(hosts.all.find((h) => h.id === "hal9000-7680")!);
    expect(Object.keys(jobs.queues)).toEqual(["hal9000-7680"]);
    expect(jobs.queues["hal9000-7680"]?.entries).toHaveLength(2);
    // Every request went to the host it was asked about.
    for (const [target] of apiJsonTo.mock.calls as [{ baseUrl: string }, string][]) {
      expect(target.baseUrl).toBe("http://hal9000:7680");
    }
  });

  it("pause and resume post against the right host", async () => {
    seedHosts();
    installApi();
    const jobs = useJobsStore();
    await jobs.refresh();
    await jobs.pause("hal9000-7680");
    let [target, path, init] = apiFetchTo.mock.lastCall as [
      { baseUrl: string },
      string,
      { method: string },
    ];
    expect(target.baseUrl).toBe("http://hal9000:7680");
    expect(path).toBe("/api/queue/pause");
    expect(init.method).toBe("POST");
    expect(jobs.queues["hal9000-7680"]?.paused).toBe(true);

    await jobs.resume("hal9000-7680");
    [target, path, init] = apiFetchTo.mock.lastCall as typeof target extends never
      ? never
      : [{ baseUrl: string }, string, { method: string }];
    expect(path).toBe("/api/queue/resume");
    expect(jobs.queues["hal9000-7680"]?.paused).toBe(false);
  });

  it("refresh() records the host's GPU ordinals for lane rendering", async () => {
    seedHosts();
    installApi({ gpus: [{ ordinal: 0 }, { ordinal: 1 }] });
    const jobs = useJobsStore();
    await jobs.refresh();
    expect(jobs.queues["local"]?.gpuOrdinals).toEqual([0, 1]);
  });

  it("refresh() leaves gpuOrdinals empty on servers that report no gpus field", async () => {
    seedHosts();
    installApi();
    const jobs = useJobsStore();
    await jobs.refresh();
    expect(jobs.queues["local"]?.gpuOrdinals).toEqual([]);
  });

  it("refresh() never creates queue lanes for non-routable worker rows", async () => {
    seedHosts();
    installApi({
      gpus: [
        { ordinal: 0, state: "degraded" },
        { ordinal: 1, state: "idle" },
      ] as never,
    });
    const jobs = useJobsStore();
    await jobs.refresh();

    expect(jobs.queues["local"]?.gpuOrdinals).toEqual([1]);
  });

  it("uses only /api/devices schedulable ordinals for current-server lanes", async () => {
    seedHosts();
    installApi({
      gpus: [
        { ordinal: 0, state: "idle" },
        { ordinal: 1, state: "idle" },
        { ordinal: 2, state: "idle" },
        { ordinal: 3, state: "idle" },
      ] as never,
    });
    listDevices.mockResolvedValue({
      plan_version: 4,
      devices: [
        device(0, {
          admin_state: "startup_excluded",
          desired_enabled: false,
          schedulable: false,
        }),
        device(1, {
          admin_state: "disabled",
          desired_enabled: false,
          schedulable: false,
        }),
        device(2, {
          health: "unavailable",
          schedulable: false,
        }),
        device(3),
      ],
    });

    const jobs = useJobsStore();
    await jobs.refresh();

    expect(jobs.queues["local"]?.gpuOrdinals).toEqual([3]);
  });

  it("reassignGpu PATCHes the owning host with a JSON target_gpu body", async () => {
    seedHosts();
    installApi({ gpus: [{ ordinal: 0 }, { ordinal: 1 }] });
    const jobs = useJobsStore();
    await jobs.refresh();
    apiJsonTo.mockClear();

    const ok = await jobs.reassignGpu("hal9000-7680", "srv-2", 1);
    expect(ok).toBe(true);
    const [target, path, init] = apiFetchTo.mock.lastCall as [
      { baseUrl: string; apiKey: string | null },
      string,
      { method: string; body: string; headers: Record<string, string> },
    ];
    expect(target).toMatchObject({ baseUrl: "http://hal9000:7680", apiKey: "hk" });
    expect(path).toBe("/api/queue/srv-2");
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body)).toEqual({ target_gpu: 1 });
    expect(init.headers).toMatchObject({ "Content-Type": "application/json" });
    // Never optimistic: server truth is re-fetched after the PATCH.
    expect(apiJsonTo.mock.calls.some(([, p]) => p === "/api/queue")).toBe(true);
  });

  it("reassignGpu on 409 (job already started) toasts an error and refetches", async () => {
    seedHosts();
    installApi({ gpus: [{ ordinal: 0 }, { ordinal: 1 }] });
    const jobs = useJobsStore();
    await jobs.refresh();
    apiFetchTo.mockRejectedValueOnce(
      Object.assign(new Error("queue job srv-2 is already running"), { status: 409 }),
    );
    apiJsonTo.mockClear();

    const ok = await jobs.reassignGpu("hal9000-7680", "srv-2", 1);
    expect(ok).toBe(false);
    const toasts = useToastStore();
    expect(toasts.items).toHaveLength(1);
    expect(toasts.items[0]).toMatchObject({ kind: "error" });
    expect(toasts.items[0]!.message).toMatch(/already started/i);
    // The store still re-syncs the host's queue so the UI shows server truth.
    expect(apiJsonTo.mock.calls.some(([, p]) => p === "/api/queue")).toBe(true);
  });

  it("reassignGpu maps 404 and 422 to directed error toasts", async () => {
    seedHosts();
    installApi();
    const jobs = useJobsStore();
    await jobs.refresh();
    const toasts = useToastStore();

    apiFetchTo.mockRejectedValueOnce(Object.assign(new Error("not found"), { status: 404 }));
    expect(await jobs.reassignGpu("local", "srv-9", 0)).toBe(false);
    expect(toasts.items.at(-1)!.message).toMatch(/no longer queued/i);

    apiFetchTo.mockRejectedValueOnce(Object.assign(new Error("bad gpu"), { status: 422 }));
    expect(await jobs.reassignGpu("local", "srv-2", 7)).toBe(false);
    expect(toasts.items.at(-1)!.message).toMatch(/not available/i);
  });

  it("cancelAll deletes the host's whole queue", async () => {
    seedHosts();
    installApi();
    const jobs = useJobsStore();
    await jobs.refresh();
    await jobs.cancelAll("local");
    const [target, path, init] = apiFetchTo.mock.lastCall as [
      { baseUrl: string },
      string,
      { method: string },
    ];
    expect(target.baseUrl).toBe("http://127.0.0.1:49152");
    expect(path).toBe("/api/queue");
    expect(init.method).toBe("DELETE");
  });
});

describe("enrichQueueEntries", () => {
  it("marks entries owned by this app and carries their client id", () => {
    const mine = { id: "srv-1", clientId: 7, hostId: null } as unknown as Job;
    const entries = [
      { id: "srv-1", model: "m", state: "running" as const, started_at_unix_ms: 1, position: 0 },
      { id: "srv-9", model: "m", state: "queued" as const, started_at_unix_ms: 2, position: 1 },
    ];
    const enriched = enrichQueueEntries(entries, "local", [mine], "local");
    expect(enriched[0]).toMatchObject({ mine: true, clientId: 7 });
    expect(enriched[1]).toMatchObject({ mine: false, clientId: null });
  });

  it("does not claim same-id entries from a different host", () => {
    const mine = { id: "srv-1", clientId: 7, hostId: "hal9000-7680" } as unknown as Job;
    const entries = [
      { id: "srv-1", model: "m", state: "queued" as const, started_at_unix_ms: 1, position: 0 },
    ];
    const enriched = enrichQueueEntries(entries, "local", [mine], "local");
    expect(enriched[0]?.mine).toBe(false);
  });
});
