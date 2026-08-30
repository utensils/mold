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

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
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

describe("held rows", () => {
  it("keeps a held row in the newest-first listing", async () => {
    // A held job exceeded its replay or dispatch cap: it exists only in the
    // journal and will never start on its own. Dropping it here is the one
    // outcome the held-row visibility contract forbids — nothing reports it,
    // and the operator cannot clear the row that is guaranteed to be stuck.
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        return Promise.resolve({
          entries: [
            {
              id: "srv-held",
              model: "flux2-klein",
              state: "held",
              started_at_unix_ms: 1,
              position: 2,
              durable: true,
              held_reason: "dispatch attempts exhausted",
            },
            {
              id: "srv-run",
              model: "flux2-klein",
              state: "running",
              started_at_unix_ms: 2,
              position: 0,
              gpu: 0,
            },
          ],
        });
      }
      if (path === "/api/status") return Promise.resolve({ queue_depth: 0 });
      if (path === "/api/capabilities") return Promise.resolve({ queue: {} });
      return Promise.resolve({});
    });
    listDevices.mockResolvedValue({ plan_version: 1, devices: [device(0)] });
    const hosts = seedHosts();
    const host = hosts.all.find((entry) => entry.id === "hal9000-7680")!;
    const jobs = useJobsStore();

    await jobs.refreshHost(host);

    const states = (jobs.queues[host.id]?.entries ?? []).map((entry) => entry.state);
    expect(states).toContain("held");
    const surface = jobs.queueSurface.map((row) => row.entry.id);
    expect(surface).toEqual(["srv-run", "srv-held"]);
    const held = jobs.queues[host.id]?.entries.find((entry) => entry.state === "held");
    expect(held?.held_reason).toBe("dispatch attempts exhausted");
  });
});

describe("jobs store", () => {
  it("uses the host queue capacity as the payload-free first page and explicitly continues", async () => {
    const hosts = seedHosts();
    const host = hosts.all.find((entry) => entry.id === "hal9000-7680")!;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status")
        return Promise.resolve({ version: "0.20.0", queue_paused: false, queue_capacity: 2 });
      if (path === "/api/capabilities") return Promise.resolve({ queue: {} });
      if (path === "/api/queue?limit=2")
        return Promise.resolve({
          entries: [
            { id: "one", model: "m", state: "running", started_at_unix_ms: 1, position: 0 },
            { id: "two", model: "m", state: "queued", started_at_unix_ms: 2, position: 1 },
          ],
          live_only_entries: [],
          page: { limit: 2, offset: 0, returned: 2, next_cursor: "page-2" },
        });
      if (path === "/api/queue?limit=2&cursor=page-2")
        return Promise.resolve({
          entries: [
            { id: "three", model: "m", state: "queued", started_at_unix_ms: 3, position: 2 },
          ],
          live_only_entries: [],
          page: { limit: 2, offset: 2, returned: 1 },
        });
      return Promise.reject(new Error(`unexpected path ${path}`));
    });
    listDevices.mockResolvedValue({ plan_version: 1, devices: [device(0)] });
    const jobs = useJobsStore();

    await jobs.refreshHost(host);
    expect(jobs.queues[host.id]?.entries.map(({ id }) => id)).toEqual(["one", "two"]);
    expect(apiJsonTo.mock.calls.some(([, path]) => path === "/api/queue")).toBe(false);

    await jobs.loadMoreHost(host.id);
    expect(jobs.queues[host.id]?.entries.map(({ id }) => id)).toEqual(["one", "two", "three"]);
    expect(jobs.queues[host.id]?.nextCursor).toBeNull();

    await jobs.refreshHost(host);
    expect(jobs.queues[host.id]?.entries.map(({ id }) => id)).toEqual(["one", "two"]);
    expect(jobs.queues[host.id]?.nextCursor).toBe("page-2");
    expect(jobs.queues[host.id]?.continued).toBe(false);
  });

  it("cannot apply a load-more failure to a replacement queue authority", async () => {
    const hosts = seedHosts();
    const host = hosts.all.find((entry) => entry.id === "hal9000-7680")!;
    const continuation = deferred<unknown>();
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve({ queue_capacity: 1 });
      if (path === "/api/capabilities") return Promise.resolve({ queue: {} });
      if (path === "/api/queue?limit=1")
        return Promise.resolve({
          entries: [
            { id: "head", model: "m", state: "queued", started_at_unix_ms: 1, position: 0 },
          ],
          page: { limit: 1, offset: 0, returned: 1, next_cursor: "next" },
        });
      if (path === "/api/queue?limit=1&cursor=next") return continuation.promise;
      return Promise.reject(new Error(`unexpected path ${path}`));
    });
    const jobs = useJobsStore();
    await jobs.refreshHost(host);
    const stale = jobs.loadMoreHost(host.id);
    await jobs.refreshHost(host);
    continuation.resolve({ error: "old host failed" });
    await stale;

    expect(jobs.queues[host.id]?.loadMoreError).toBeNull();
    expect(jobs.queues[host.id]?.loadingMore).toBe(false);
  });

  it("self-schedules polling only after the previous refresh settles", async () => {
    vi.useFakeTimers();
    seedHosts();
    const jobs = useJobsStore();
    const first = deferred<void>();
    const refresh = vi.spyOn(jobs, "refresh").mockReturnValueOnce(first.promise);

    jobs.startPolling();
    await Promise.resolve();
    expect(refresh).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(20_000);
    expect(refresh).toHaveBeenCalledTimes(1);

    refresh.mockResolvedValue(undefined);
    first.resolve();
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(5_001);
    expect(refresh).toHaveBeenCalledTimes(2);
    jobs.stopPolling();
    vi.useRealTimers();
  });

  it("refresh() snapshots every ready host's queue, pause state, and capabilities", async () => {
    seedHosts();
    installApi({ paused: true });
    const jobs = useJobsStore();
    await jobs.refresh();
    expect(Object.keys(jobs.queues).sort()).toEqual(["hal9000-7680", "local"]);
    const q = jobs.queues["local"]!;
    expect(q.entries).toHaveLength(2);
    expect(q.paused).toBe(true);
    expect(q.caps).toEqual({
      canPause: true,
      canCancelAll: true,
      canReorder: true,
      canCancelRunning: false,
    });
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

  it("ignores an older same-host queue refresh after a newer refresh settles", async () => {
    const hosts = seedHosts();
    const host = hosts.all.find((entry) => entry.id === "hal9000-7680")!;
    const older = deferred<unknown>();
    const newer = deferred<unknown>();
    let queueCall = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") return queueCall++ === 0 ? older.promise : newer.promise;
      if (path === "/api/status")
        return Promise.resolve({ version: "0.20.0", queue_paused: false });
      return Promise.reject(new Error(`unexpected path ${path}`));
    });
    listDevices.mockResolvedValue({ plan_version: 1, devices: [device(0)] });
    const jobs = useJobsStore();
    jobs.queues[host.id] = {
      hostId: host.id,
      entries: [],
      paused: false,
      caps: { canPause: true, canCancelAll: true, canReorder: true },
      gpuOrdinals: [],
      devices: [],
      plan: null,
      error: null,
    };

    const first = jobs.refreshHost(host);
    const second = jobs.refreshHost(host);
    newer.resolve({
      entries: [
        {
          id: "newer",
          model: "flux-dev:q4",
          state: "queued",
          started_at_unix_ms: 2,
          position: 0,
        },
      ],
    });
    await second;
    older.resolve({
      entries: [
        {
          id: "older",
          model: "flux-dev:q4",
          state: "queued",
          started_at_unix_ms: 1,
          position: 0,
        },
      ],
    });
    await first;

    expect(jobs.queues[host.id]?.entries.map((entry) => entry.id)).toEqual(["newer"]);
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

  it("recognizes a durable client batch after the app restarts", () => {
    const entries = [
      {
        id: "srv-held",
        model: "m",
        state: "held" as const,
        started_at_unix_ms: 1,
        position: 0,
        client_batch_id: "client-recovered",
      },
    ];
    const enriched = enrichQueueEntries(
      entries,
      "local",
      [],
      "local",
      (clientBatchId) => clientBatchId === "client-recovered",
    );
    expect(enriched[0]).toMatchObject({ mine: true, clientId: null });
  });
});
