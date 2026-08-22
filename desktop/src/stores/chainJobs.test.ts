import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import type { ApiTarget } from "../lib/api/client";
import type { ChainRequestWire } from "@studio/lib/api/chainTypes";

const { apiFetchTo, apiJsonTo, sseStream } = vi.hoisted(() => ({
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(),
  sseStream: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/api/client", () => ({
  apiFetch: vi.fn(),
  apiFetchTo,
  apiJson: vi.fn(),
  apiJsonTo,
}));
vi.mock("../lib/api/sse", () => ({ sseStream }));
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn().mockResolvedValue({}),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
  },
}));
vi.mock("../lib/notify", () => ({ notifyChainFinished: vi.fn() }));

import { useChainJobsStore } from "./chainJobs";
import { useConnectionStore } from "./connection";
import { useHostsStore } from "./hosts";

const LOCAL_TARGET: ApiTarget = { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
const REMOTE_TARGET: ApiTarget = { baseUrl: "http://hal9000:7680", apiKey: "remote-key" };
const REMOTE_ID = "hal9000-7680";

const request = { model: "ltx-video", stages: [] } as unknown as ChainRequestWire;

function readyHosts() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: LOCAL_TARGET.baseUrl, apiKey: LOCAL_TARGET.apiKey };
  conn.status = "ready";
  const hosts = useHostsStore();
  hosts.initialized = true;
  hosts.extras.push({
    id: REMOTE_ID,
    label: "hal9000",
    url: REMOTE_TARGET.baseUrl,
    apiKey: REMOTE_TARGET.apiKey,
    status: "ready",
    error: null,
    instanceId: null,
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  apiJsonTo.mockImplementation((_target: ApiTarget, path: string, init?: RequestInit) =>
    Promise.resolve(
      path === "/api/chain-jobs" && init?.method === "POST"
        ? { job_id: "remote-chain-1" }
        : { jobs: [] },
    ),
  );
  apiFetchTo.mockImplementation((_target: ApiTarget, path: string, init?: RequestInit) =>
    Promise.resolve(
      path === "/api/chain-jobs" && init?.method === "POST"
        ? new Response(JSON.stringify({ job_id: "remote-chain-1" }))
        : {},
    ),
  );
  readyHosts();
});

describe("per-host listing", () => {
  it("fetchHost stores each host's jobs under its own id, newest first", async () => {
    apiJsonTo.mockImplementation((target: ApiTarget) =>
      Promise.resolve({
        jobs:
          target.baseUrl === REMOTE_TARGET.baseUrl
            ? [
                { id: "r-old", state: "completed", created_at_unix_ms: 1 },
                { id: "r-new", state: "completed", created_at_unix_ms: 2 },
              ]
            : [{ id: "l1", state: "completed", created_at_unix_ms: 5 }],
      }),
    );
    const store = useChainJobsStore();
    await store.fetchAll();
    expect(store.byHost.local?.jobs.map((j) => j.id)).toEqual(["l1"]);
    expect(store.byHost[REMOTE_ID]?.jobs.map((j) => j.id)).toEqual(["r-new", "r-old"]);
  });

  it("keeps automatic long-video shim jobs out of sequence surfaces", async () => {
    apiJsonTo.mockResolvedValue({
      jobs: [
        { id: "authored", state: "running", created_at_unix_ms: 1, ephemeral: false },
        { id: "one-shot", state: "running", created_at_unix_ms: 2, ephemeral: true },
      ],
    });
    const store = useChainJobsStore();
    await store.fetchHost("local");

    expect(store.byHost.local?.jobs.map((job) => job.id)).toEqual(["authored"]);
  });

  it("a failed host keeps its last jobs and records the error", async () => {
    const store = useChainJobsStore();
    store.byHost[REMOTE_ID] = {
      jobs: [{ id: "r1", state: "completed" }] as never,
      error: null,
    };
    apiJsonTo.mockRejectedValue(new Error("boom"));
    await store.fetchHost(REMOTE_ID);
    expect(store.byHost[REMOTE_ID]?.jobs.map((j) => j.id)).toEqual(["r1"]);
    expect(store.byHost[REMOTE_ID]?.error).toContain("boom");
  });

  it("fetching one host never clobbers a watch on another host", async () => {
    const store = useChainJobsStore();
    store.watch("local", "local-chain");
    const abort = store.abort!;
    await store.fetchHost(REMOTE_ID);
    expect(store.watching).toEqual({ hostId: "local", jobId: "local-chain" });
    expect(abort.signal.aborted).toBe(false);
  });
});

describe("create", () => {
  it("creates on the addressed host, refreshes it, and watches through the same target", async () => {
    const store = useChainJobsStore();
    const jobId = await store.create(REMOTE_ID, request);
    expect(jobId).toBe("remote-chain-1");
    expect(apiFetchTo).toHaveBeenCalledWith(
      REMOTE_TARGET,
      "/api/chain-jobs",
      expect.objectContaining({ method: "POST" }),
    );
    expect(apiJsonTo).toHaveBeenCalledWith(REMOTE_TARGET, "/api/chain-jobs");
    expect(sseStream).toHaveBeenCalledWith(
      "/api/chain-jobs/remote-chain-1/events",
      expect.objectContaining({ target: REMOTE_TARGET }),
    );
  });

  it("stops a previous host's watch before creating on a different host", async () => {
    const store = useChainJobsStore();
    store.watch("local", "local-chain");
    const previousAbort = store.abort!;
    await store.create(REMOTE_ID, request);
    expect(previousAbort.signal.aborted).toBe(true);
    expect(store.watching).toEqual({ hostId: REMOTE_ID, jobId: "remote-chain-1" });
  });
});

describe("per-host actions resolve their target at call time", () => {
  it("cancelling a host-B job hits host B even after fetching host A", async () => {
    const store = useChainJobsStore();
    await store.fetchHost("local");
    await store.cancel(REMOTE_ID, "r1");
    expect(apiFetchTo).toHaveBeenCalledWith(REMOTE_TARGET, "/api/chain-jobs/r1/cancel", {
      method: "POST",
    });
  });

  it("remove targets the owning host and unwatches a removed watched job", async () => {
    const store = useChainJobsStore();
    store.watch(REMOTE_ID, "r1");
    store.byHost[REMOTE_ID] = {
      jobs: [{ id: "r1", state: "completed" }] as never,
      error: null,
    };
    await store.remove(REMOTE_ID, "r1");
    expect(apiFetchTo).toHaveBeenCalledWith(REMOTE_TARGET, "/api/chain-jobs/r1", {
      method: "DELETE",
    });
    expect(store.watching).toBeNull();
    expect(store.byHost[REMOTE_ID]?.jobs).toEqual([]);
  });

  it("resume and retake re-watch on the owning host", async () => {
    const store = useChainJobsStore();
    await store.resume(REMOTE_ID, "r1");
    expect(apiFetchTo).toHaveBeenCalledWith(REMOTE_TARGET, "/api/chain-jobs/r1/resume", {
      method: "POST",
    });
    expect(store.watching).toEqual({ hostId: REMOTE_ID, jobId: "r1" });

    await store.retake("local", "l1", { stage_idx: 0, mode: "cascade" });
    expect(apiFetchTo).toHaveBeenCalledWith(
      LOCAL_TARGET,
      "/api/chain-jobs/l1/retake",
      expect.objectContaining({ method: "POST" }),
    );
    expect(store.watching).toEqual({ hostId: "local", jobId: "l1" });
  });

  it("amend POSTs the edited stages to the owning host and re-watches", async () => {
    const store = useChainJobsStore();
    apiJsonTo.mockImplementation((_t: ApiTarget, path: string, init?: RequestInit) =>
      Promise.resolve(
        init?.method === "POST" && path.endsWith("/amend")
          ? { id: "r1", state: "queued", preserved_stages: 1 }
          : { jobs: [] },
      ),
    );
    const outcome = await store.amend(REMOTE_ID, "r1", { stages: [] });
    expect(outcome.preserved_stages).toBe(1);
    expect(apiJsonTo).toHaveBeenCalledWith(
      REMOTE_TARGET,
      "/api/chain-jobs/r1/amend",
      expect.objectContaining({ method: "POST" }),
    );
    expect(store.watching).toEqual({ hostId: REMOTE_ID, jobId: "r1" });
  });

  it("gc runs on the addressed host", async () => {
    const store = useChainJobsStore();
    apiJsonTo.mockImplementation((_t: ApiTarget, path: string) =>
      Promise.resolve(
        path === "/api/chain-jobs/gc"
          ? { swept_ephemeral_jobs: 2, pruned_artifact_dirs: 1 }
          : { jobs: [] },
      ),
    );
    const outcome = await store.gc(REMOTE_ID);
    expect(outcome.swept_ephemeral_jobs).toBe(2);
    expect(apiJsonTo).toHaveBeenCalledWith(REMOTE_TARGET, "/api/chain-jobs/gc", {
      method: "POST",
    });
  });
});

describe("clearInactive", () => {
  it("deletes every inactive job on every host, resumable included", async () => {
    const store = useChainJobsStore();
    store.byHost.local = {
      jobs: [
        { id: "a", state: "completed" },
        { id: "b", state: "running" },
      ] as never,
      error: null,
    };
    store.byHost[REMOTE_ID] = {
      jobs: [
        { id: "c", state: "failed" },
        { id: "d", state: "queued" },
        { id: "e", state: "cancelled" },
      ] as never,
      error: null,
    };

    const outcome = await store.clearInactive();

    expect(outcome).toEqual({ cleared: 3, failed: 0 });
    expect(apiFetchTo).toHaveBeenCalledWith(LOCAL_TARGET, "/api/chain-jobs/a", {
      method: "DELETE",
    });
    expect(apiFetchTo).toHaveBeenCalledWith(REMOTE_TARGET, "/api/chain-jobs/c", {
      method: "DELETE",
    });
    expect(apiFetchTo).toHaveBeenCalledWith(REMOTE_TARGET, "/api/chain-jobs/e", {
      method: "DELETE",
    });
    expect(store.byHost.local?.jobs.map((j) => j.id)).toEqual(["b"]);
    expect(store.byHost[REMOTE_ID]?.jobs.map((j) => j.id)).toEqual(["d"]);
  });

  it("scopes to one host when given an id and resyncs on partial failure", async () => {
    const store = useChainJobsStore();
    store.byHost.local = { jobs: [{ id: "a", state: "completed" }] as never, error: null };
    store.byHost[REMOTE_ID] = {
      jobs: [
        { id: "c", state: "failed" },
        { id: "e", state: "completed" },
      ] as never,
      error: null,
    };
    apiFetchTo.mockRejectedValueOnce(new Error("boom")).mockResolvedValue({});

    const outcome = await store.clearInactive(REMOTE_ID);

    expect(outcome.cleared + outcome.failed).toBe(2);
    expect(outcome.failed).toBe(1);
    expect(apiFetchTo).not.toHaveBeenCalledWith(LOCAL_TARGET, "/api/chain-jobs/a", {
      method: "DELETE",
    });
    // Partial failures resync the host from the server rather than guessing.
    expect(apiJsonTo).toHaveBeenCalledWith(REMOTE_TARGET, "/api/chain-jobs");
  });
});

describe("watch stream", () => {
  it("reduces snapshot events into the live view and bumps finalizedTick", async () => {
    const store = useChainJobsStore();
    store.watch(REMOTE_ID, "r1");
    const onEvent = sseStream.mock.calls[0]![1].onEvent as (e: string, d: string) => void;
    onEvent(
      "chain_job",
      JSON.stringify({
        type: "snapshot",
        job: {
          id: "r1",
          state: "running",
          model: "ltx-video",
          stage_count: 2,
          current_stage: 0,
          created_at_unix_ms: 1,
          updated_at_unix_ms: 1,
          stages: [
            { idx: 0, state: "running" },
            { idx: 1, state: "pending" },
          ],
        },
      }),
    );
    expect(store.live.detail?.id).toBe("r1");
    expect(store.live.activeStage).toBe(0);

    expect(store.finalizedTick).toBe(0);
    onEvent("chain_job", JSON.stringify({ type: "finalized", output: "out.mp4", take: 1 }));
    expect(store.finalizedTick).toBe(1);
  });

  it("ignores frames from a superseded watch", () => {
    const store = useChainJobsStore();
    store.watch(REMOTE_ID, "r1");
    const onEvent = sseStream.mock.calls[0]![1].onEvent as (e: string, d: string) => void;
    store.watch("local", "l1");
    onEvent("chain_job", JSON.stringify({ type: "finalized" }));
    expect(store.finalizedTick).toBe(0);
  });
});
