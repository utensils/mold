import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

const apiJsonTo = vi.fn();
const apiFetchTo = vi.fn().mockResolvedValue(new Response(null, { status: 200 }));
vi.mock("../lib/api/client", () => ({
  apiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
  apiFetchTo: (...a: unknown[]) => apiFetchTo(...a),
}));

import { useConnectionStore } from "./connection";
import { useHostsStore } from "./hosts";
import { enrichQueueEntries, useJobsStore } from "./jobs";
import type { Job } from "./generation";

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
function installApi({ paused = false }: { paused?: boolean } = {}) {
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
    if (path === "/api/status") return Promise.resolve({ version: "0.17.0", queue_paused: paused });
    if (path === "/api/capabilities") {
      return Promise.resolve({ queue: { can_pause: true, can_cancel_all: true } });
    }
    return Promise.reject(new Error(`unexpected path ${path}`));
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  apiFetchTo.mockResolvedValue(new Response(null, { status: 200 }));
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
    expect(q.caps).toEqual({ canPause: true, canCancelAll: true });
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
