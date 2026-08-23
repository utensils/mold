import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../lib/api/sse", () => ({ sseStream: vi.fn() }));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJsonTo: vi.fn(),
  apiFetchTo: vi.fn(() => Promise.resolve(new Response(null, { status: 204 }))),
}));
const effectMocks = vi.hoisted(() => ({
  notifyGenerated: vi.fn(),
  notifyGenerationFailed: vi.fn(),
  fetchGalleryMediaBytes: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
  saveOutputBytes: vi.fn().mockResolvedValue("saved.png"),
}));
vi.mock("../lib/notify", () => ({
  notifyGenerated: effectMocks.notifyGenerated,
  notifyGenerationFailed: effectMocks.notifyGenerationFailed,
}));
vi.mock("../lib/gallery/media", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/gallery/media")>()),
  streamableMediaUrl: vi.fn().mockResolvedValue("blob:durable-result"),
  fetchGalleryMediaBytes: effectMocks.fetchGalleryMediaBytes,
}));
vi.mock("../lib/ipc", () => ({
  ipc: { saveOutputBytes: effectMocks.saveOutputBytes },
}));
const durableApi = vi.hoisted(() => ({
  admit: vi.fn(),
  lookup: vi.fn(),
  reconcile: vi.fn(),
}));
vi.mock("@studio/api/generationAdmission", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/generationAdmission")>()),
  admitGenerationBatch: (...args: unknown[]) => durableApi.admit(...args),
  lookupGenerationBatchByClientId: (...args: unknown[]) => durableApi.lookup(...args),
  reconcileGenerationBatches: (...args: unknown[]) => durableApi.reconcile(...args),
}));

import { sseStream } from "../lib/api/sse";
import { apiFetchTo, apiJsonTo } from "../lib/api/client";
import { runWithConcurrency, useGenerationStore } from "./generation";
import { useHostsStore } from "./hosts";
import type { GenerateRequest } from "../lib/api/types";
import { DURABLE_GENERATION_STORAGE_KEY } from "../lib/durableGeneration";
import {
  createGenerationBatchTracker,
  reduceGenerationLifecycle,
} from "@studio/lib/generationLifecycle";

describe("runWithConcurrency", () => {
  it("never exceeds the concurrency limit", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const releases: Array<() => void> = [];
    const tasks = Array.from({ length: 6 }, () => () => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      return new Promise<void>((resolve) => {
        releases.push(() => {
          inFlight--;
          resolve();
        });
      });
    });

    const done = runWithConcurrency(tasks, 2);
    await flushPromises();
    expect(inFlight).toBe(2);

    // Drain oldest-first; each release frees a slot for the next task.
    while (releases.length) {
      releases.shift()!();
      await flushPromises();
    }
    await done;

    expect(maxInFlight).toBe(2);
    expect(inFlight).toBe(0);
  });

  it("resolves results in task order regardless of completion order", async () => {
    const tasks = [
      () => new Promise<string>((r) => setTimeout(() => r("a"), 30)),
      () => Promise.resolve("b"),
      () => new Promise<string>((r) => setTimeout(() => r("c"), 5)),
    ];
    expect(await runWithConcurrency(tasks, 2)).toEqual(["a", "b", "c"]);
  });

  it("a rejected task does not stall its siblings", async () => {
    const tasks = [
      () => Promise.reject(new Error("boom")),
      () => Promise.resolve("b"),
      () => Promise.resolve("c"),
    ];
    const results = await runWithConcurrency(tasks, 2);
    expect(results[1]).toBe("b");
    expect(results[2]).toBe("c");
  });
});

describe("submitBatch connection cap", () => {
  const mockSse = vi.mocked(sseStream);
  const streams: Array<{
    seed: number;
    target: string;
    onEvent: (event: string, data: string) => void;
    resolve: () => void;
  }> = [];

  function resolveStream(seed: number) {
    const idx = streams.findIndex((c) => c.seed === seed);
    streams[idx]!.resolve();
  }

  function completeStream(seed: number) {
    const stream = streams.find((candidate) => candidate.seed === seed);
    stream!.onEvent(
      "complete",
      JSON.stringify({
        image: btoa("generated"),
        format: "png",
        width: 1024,
        height: 1024,
        seed_used: seed,
        generation_time_ms: 100,
        model: req.model,
      }),
    );
  }

  const req: GenerateRequest = {
    prompt: "a lighthouse",
    model: "flux-schnell:q8",
    width: 1024,
    height: 1024,
    steps: 4,
    seed: 100,
  };

  beforeEach(() => {
    setActivePinia(createPinia());
    streams.length = 0;
    mockSse.mockReset();
    vi.mocked(apiJsonTo).mockReset();
    vi.mocked(apiFetchTo).mockReset();
    vi.mocked(apiFetchTo).mockResolvedValue(new Response(null, { status: 204 }));
    durableApi.admit.mockReset();
    durableApi.lookup.mockReset();
    durableApi.reconcile.mockReset();
    effectMocks.notifyGenerated.mockClear();
    effectMocks.notifyGenerationFailed.mockClear();
    effectMocks.fetchGalleryMediaBytes.mockClear();
    effectMocks.saveOutputBytes.mockClear();
    // Each POST parks open until the test resolves it, so we can observe how
    // many held streams the batch opens at once.
    mockSse.mockImplementation((_url, opts) => {
      return new Promise<void>((resolve) => {
        let settled = false;
        const stream = {
          seed: (opts.body as { seed: number }).seed,
          target: opts.target?.baseUrl ?? "__primary__",
          onEvent: opts.onEvent,
          resolve: () => {},
        };
        const finish = () => {
          if (settled) return;
          settled = true;
          opts.signal.removeEventListener("abort", finish);
          const index = streams.indexOf(stream);
          if (index >= 0) streams.splice(index, 1);
          resolve();
        };
        stream.resolve = finish;
        opts.signal.addEventListener("abort", finish, { once: true });
        streams.push(stream);
      });
    });
  });

  afterEach(() => {
    const store = useGenerationStore();
    store.resetJobs();
    vi.unstubAllGlobals();
  });

  it("holds at most four sibling streams open at once", async () => {
    const store = useGenerationStore();
    store.submitBatch(req, 5);
    await flushPromises();
    expect(streams.map((s) => s.seed).sort()).toEqual([100, 101, 102, 103]);

    resolveStream(100);
    await flushPromises();
    expect(streams.map((s) => s.seed).sort()).toEqual([101, 102, 103, 104]);
  });

  it("retries one committed heterogeneous admission with the same UUID and maps all 30 ids", async () => {
    const calls: Array<{ path: string; body: string | undefined }> = [];
    let admissions = 0;
    vi.mocked(apiJsonTo).mockImplementation(async (_target, path, init) => {
      calls.push({ path, body: typeof init?.body === "string" ? init.body : undefined });
      if (path === "/api/generation-batches") {
        admissions += 1;
        if (admissions === 1) throw new TypeError("Load failed");
        return {
          batch_id: "server-batch",
          client_batch_id: "client-batch",
          state: "queued",
          created_at_ms: 1,
          updated_at_ms: 1,
          children: Array.from({ length: 30 }, (_, index) => ({
            index: index + 1,
            job_id: `job-${index + 1}`,
            state: "queued",
            error: null,
          })),
        } as never;
      }
      return new Promise<never>(() => {});
    });
    const store = useGenerationStore();
    const { jobs } = store.submitBatch(
      req,
      30,
      {
        hostId: "hal9000",
        label: "hal9000",
        kind: "remote",
        target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
        heterogeneousBatch: true,
        heterogeneousBatchMaxOutputs: 64,
      },
      null,
      { batchId: "client-batch" },
    );
    await flushPromises();
    await flushPromises();

    const admissionCalls = calls.filter((call) => call.path === "/api/generation-batches");
    expect(admissionCalls).toHaveLength(2);
    expect(admissionCalls[1]!.body).toBe(admissionCalls[0]!.body);
    expect(JSON.parse(admissionCalls[0]!.body!).client_batch_id).toBe("client-batch");
    expect(jobs.map((job) => job.id)).toEqual(
      Array.from({ length: 30 }, (_, index) => `job-${index + 1}`),
    );
    expect(mockSse).not.toHaveBeenCalled();
  });

  it("admits singleton and Batch N durably without held generation streams", async () => {
    const store = useGenerationStore();
    const hosts = useHostsStore();
    hosts.extras = [
      {
        id: "hal9000",
        label: "hal9000",
        url: "http://hal9000:7680",
        apiKey: "fresh-key",
        status: "ready",
        error: null,
        instanceId: "instance-1",
      },
    ];
    store.attachSharedDurableEventHost("hal9000");
    let nextBatch = 0;
    durableApi.admit.mockImplementation(async (_target, body) => {
      nextBatch += 1;
      const requestBody = body as {
        client_batch_id: string;
        requests: GenerateRequest[];
      };
      return {
        id: `batch-${nextBatch}`,
        client_batch_id: requestBody.client_batch_id,
        instance_id: "instance-1",
        durable: true,
        children: requestBody.requests.map((_request, index) => ({
          index: index + 1,
          job_id: `job-${nextBatch}-${index + 1}`,
          state: "queued",
          created_at_ms: 1,
          updated_at_ms: 1,
        })),
      };
    });
    const route = {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote" as const,
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatch: true,
      heterogeneousBatchMaxOutputs: 64,
      durableBatchOutcomes: true,
      mirrorRemoteOutput: false,
    };

    const singleton = store.submitBatch(req, 1, route);
    const batch = store.submitBatch(req, 5, route);
    await flushPromises();

    expect(singleton.jobs.map((job) => job.id)).toEqual(["job-1-1"]);
    expect(batch.jobs.map((job) => job.id)).toEqual([
      "job-2-1",
      "job-2-2",
      "job-2-3",
      "job-2-4",
      "job-2-5",
    ]);
    expect(mockSse).not.toHaveBeenCalled();
  });

  it("recovers an ambiguous durable POST by client id without legacy fallback", async () => {
    const store = useGenerationStore();
    const hosts = useHostsStore();
    hosts.extras = [
      {
        id: "hal9000",
        label: "hal9000",
        url: "http://hal9000:7680",
        apiKey: "fresh-key",
        status: "ready",
        error: null,
        instanceId: "instance-1",
      },
    ];
    store.attachSharedDurableEventHost("hal9000");
    let clientBatchId = "";
    durableApi.admit.mockImplementation(async (_target, body) => {
      clientBatchId = (body as { client_batch_id: string }).client_batch_id;
      throw new TypeError("response lost");
    });
    durableApi.lookup.mockImplementation(async () => ({
      kind: "found",
      batch: {
        id: "batch-recovered",
        client_batch_id: clientBatchId,
        instance_id: "instance-1",
        durable: true,
        children: [
          {
            index: 1,
            job_id: "job-recovered",
            state: "queued",
            created_at_ms: 1,
            updated_at_ms: 1,
          },
        ],
      },
    }));

    const submitted = store.submitBatch(req, 1, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatch: true,
      durableBatchOutcomes: true,
      mirrorRemoteOutput: false,
    });
    await flushPromises();
    await flushPromises();

    expect(submitted.jobs[0]!.id).toBe("job-recovered");
    expect(mockSse).not.toHaveBeenCalled();
    expect(durableApi.admit).toHaveBeenCalledTimes(1);
    expect(durableApi.lookup).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://hal9000:7680" }),
      clientBatchId,
    );
  });

  it("uses host events as hints and bulk status as the only terminal authority", async () => {
    const store = useGenerationStore();
    useHostsStore().extras = [
      {
        id: "hal9000",
        label: "hal9000",
        url: "http://hal9000:7680",
        apiKey: "fresh-key",
        status: "ready",
        error: null,
        instanceId: "instance-1",
      },
    ];
    store.attachSharedDurableEventHost("hal9000");
    let clientBatchId = "";
    let phase: "running" | "complete" = "running";
    const status = () => ({
      id: "batch-1",
      client_batch_id: clientBatchId,
      instance_id: "instance-1",
      durable: true,
      children: [
        {
          index: 1,
          job_id: "job-1",
          state: phase,
          created_at_ms: 1,
          updated_at_ms: phase === "running" ? 2 : 3,
          ...(phase === "complete"
            ? {
                completed_at_ms: 3,
                result: { filename: "finished.png" },
              }
            : {}),
        },
      ],
    });
    durableApi.admit.mockImplementation(async (_target, body) => {
      clientBatchId = (body as { client_batch_id: string }).client_batch_id;
      return { ...status(), children: [{ ...status().children[0], state: "queued" }] };
    });
    durableApi.reconcile.mockImplementation(async () => ({
      instance_id: "instance-1",
      batches: [status()],
      missing: { client_batch_ids: [], batch_ids: [] },
    }));
    const submitted = store.submitBatch(req, 1, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatch: true,
      durableBatchOutcomes: true,
      mirrorRemoteOutput: true,
    });
    await flushPromises();

    store.onDurableEvent("hal9000", "authority", '{"instance_id":"instance-1"}');
    await flushPromises();
    expect(submitted.jobs[0]!.status).toBe("loading");

    phase = "complete";
    store.onDurableEvent("hal9000", "event", '{"type":"job_ended","id":"job-1"}');
    await submitted.settled;
    expect(submitted.jobs[0]).toMatchObject({
      status: "complete",
      id: "job-1",
      result: { filename: "finished.png", image: "" },
    });
    await flushPromises();
    store.onDurableEvent("hal9000", "event", '{"type":"job_ended","id":"job-1"}');
    await flushPromises();
    expect(effectMocks.notifyGenerated).toHaveBeenCalledTimes(1);
    expect(effectMocks.fetchGalleryMediaBytes).toHaveBeenCalledTimes(1);
    expect(effectMocks.fetchGalleryMediaBytes).toHaveBeenCalledWith(
      "/api/gallery/image/finished.png",
      { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
    );
    expect(effectMocks.saveOutputBytes).toHaveBeenCalledTimes(1);
    expect(mockSse).not.toHaveBeenCalled();
  });

  it("restores running durable work and accepts a newer queued snapshot after restart", async () => {
    const storage = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => void storage.set(key, value),
    });
    let tracker = createGenerationBatchTracker({
      hostId: "hal9000",
      expectedInstanceId: "instance-1",
      clientBatchId: "client-restored",
      submittedAtMs: 1,
    });
    tracker = reduceGenerationLifecycle(tracker, {
      type: "batch_snapshot",
      batch: {
        id: "batch-restored",
        client_batch_id: "client-restored",
        instance_id: "instance-1",
        durable: true,
        children: [
          {
            index: 1,
            job_id: "job-restored",
            state: "running",
            created_at_ms: 1,
            updated_at_ms: 20,
          },
        ],
      },
    });
    let staleTracker = createGenerationBatchTracker({
      hostId: "hal9000",
      expectedInstanceId: "retired-instance",
      clientBatchId: "client-stale",
      submittedAtMs: 1,
    });
    staleTracker = reduceGenerationLifecycle(staleTracker, {
      type: "batch_snapshot",
      batch: {
        id: "batch-stale",
        client_batch_id: "client-stale",
        instance_id: "retired-instance",
        durable: true,
        children: [
          {
            index: 1,
            job_id: "job-stale",
            state: "queued",
            created_at_ms: 1,
            updated_at_ms: 1,
          },
        ],
      },
    });
    let terminalTracker = createGenerationBatchTracker({
      hostId: "hal9000",
      expectedInstanceId: "retired-instance",
      clientBatchId: "client-terminal",
      submittedAtMs: 1,
    });
    terminalTracker = reduceGenerationLifecycle(terminalTracker, {
      type: "batch_snapshot",
      batch: {
        id: "batch-terminal",
        client_batch_id: "client-terminal",
        instance_id: "retired-instance",
        durable: true,
        children: [
          {
            index: 1,
            job_id: "job-terminal",
            state: "complete",
            created_at_ms: 1,
            updated_at_ms: 2,
            completed_at_ms: 2,
            result: { filename: "terminal-on-retired-host.png" },
          },
        ],
      },
    });
    const child = {
      index: 1,
      clientId: null,
      model: req.model,
      width: req.width,
      height: req.height,
      steps: req.steps,
      guidance: 1,
      seed: req.seed,
      format: "png",
    };
    storage.set(
      DURABLE_GENERATION_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        records: [
          {
            tracker,
            hostLabel: "hal9000",
            hostKind: "remote",
            mirrorRemoteOutput: false,
            children: [child],
            effectReceipts: [],
          },
          {
            tracker: staleTracker,
            hostLabel: "old hal9000",
            hostKind: "remote",
            mirrorRemoteOutput: false,
            children: [child],
            effectReceipts: [],
          },
          {
            tracker: terminalTracker,
            hostLabel: "old hal9000",
            hostKind: "remote",
            mirrorRemoteOutput: true,
            children: [child],
            effectReceipts: [],
          },
        ],
      }),
    );
    useHostsStore().extras = [
      {
        id: "hal9000",
        label: "hal9000",
        url: "http://hal9000:7680",
        apiKey: "fresh-key",
        status: "ready",
        error: null,
        instanceId: "instance-1",
      },
    ];
    durableApi.reconcile.mockResolvedValue({
      instance_id: "instance-1",
      batches: [
        {
          id: "batch-restored",
          client_batch_id: "client-restored",
          instance_id: "instance-1",
          durable: true,
          children: [
            {
              index: 1,
              job_id: "job-restored",
              state: "queued",
              created_at_ms: 1,
              updated_at_ms: 30,
            },
          ],
        },
      ],
      missing: { client_batch_ids: [], batch_ids: [] },
    });

    const store = useGenerationStore();
    store.attachSharedDurableEventHost("hal9000");
    store.resumeDurableGenerations();
    expect(store.jobs[0]).toMatchObject({ id: "job-restored", status: "loading" });
    await flushPromises();
    expect(store.jobs[0]).toMatchObject({ id: "job-restored", status: "queued" });
    expect(store.jobs[1]).toMatchObject({
      id: "job-stale",
      status: "queued",
      interrupted: true,
      stage: "Original machine identity changed — outcome unknown",
    });
    expect(store.jobs[2]).toMatchObject({
      id: "job-terminal",
      status: "complete",
      result: { filename: "terminal-on-retired-host.png" },
    });
    expect(effectMocks.fetchGalleryMediaBytes).not.toHaveBeenCalled();
    expect(storage.get(DURABLE_GENERATION_STORAGE_KEY)).toContain("client-terminal");
    expect(durableApi.reconcile).toHaveBeenCalledTimes(1);
    expect(durableApi.reconcile).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      expect.objectContaining({
        client_batch_ids: [],
        batch_ids: ["batch-restored"],
      }),
    );
  });

  it("lets a durable completion win a concurrent cancellation request", async () => {
    const store = useGenerationStore();
    useHostsStore().extras = [
      {
        id: "hal9000",
        label: "hal9000",
        url: "http://hal9000:7680",
        apiKey: "fresh-key",
        status: "ready",
        error: null,
        instanceId: "instance-1",
      },
    ];
    store.attachSharedDurableEventHost("hal9000");
    let clientBatchId = "";
    durableApi.admit.mockImplementation(async (_target, body) => {
      clientBatchId = (body as { client_batch_id: string }).client_batch_id;
      return {
        id: "batch-race",
        client_batch_id: clientBatchId,
        instance_id: "instance-1",
        durable: true,
        children: [
          {
            index: 1,
            job_id: "job-race",
            state: "queued",
            created_at_ms: 1,
            updated_at_ms: 1,
          },
        ],
      };
    });
    durableApi.reconcile.mockImplementation(async () => ({
      instance_id: "instance-1",
      batches: [
        {
          id: "batch-race",
          client_batch_id: clientBatchId,
          instance_id: "instance-1",
          durable: true,
          children: [
            {
              index: 1,
              job_id: "job-race",
              state: "complete",
              created_at_ms: 1,
              updated_at_ms: 2,
              completed_at_ms: 2,
              result: { filename: "race-winner.png" },
            },
          ],
        },
      ],
      missing: { client_batch_ids: [], batch_ids: [] },
    }));
    const submitted = store.submitBatch(req, 1, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatch: true,
      durableBatchOutcomes: true,
      mirrorRemoteOutput: false,
    });
    await flushPromises();

    await expect(store.cancel(submitted.jobs[0]!.clientId)).resolves.toBe(false);
    expect(submitted.jobs[0]).toMatchObject({
      status: "complete",
      error: null,
      result: { filename: "race-winner.png" },
    });
    expect(vi.mocked(apiFetchTo)).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      "/api/queue/job-race",
      { method: "DELETE" },
    );
  });

  it("keeps repeated durable submissions off the held-stream pool", async () => {
    const store = useGenerationStore();
    useHostsStore().extras = [
      {
        id: "hal9000",
        label: "hal9000",
        url: "http://hal9000:7680",
        apiKey: "fresh-key",
        status: "ready",
        error: null,
        instanceId: "instance-1",
      },
    ];
    store.attachSharedDurableEventHost("hal9000");
    let admissions = 0;
    durableApi.admit.mockImplementation(async (_target, body) => {
      admissions += 1;
      const client = (body as { client_batch_id: string }).client_batch_id;
      return {
        id: `batch-${admissions}`,
        client_batch_id: client,
        instance_id: "instance-1",
        durable: true,
        children: [
          {
            index: 1,
            job_id: `job-${admissions}`,
            state: "queued",
            created_at_ms: admissions,
            updated_at_ms: admissions,
          },
        ],
      };
    });
    const route = {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote" as const,
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatch: true,
      durableBatchOutcomes: true,
      mirrorRemoteOutput: false,
    };

    for (let index = 0; index < 40; index += 1) {
      store.submitBatch({ ...req, seed: index }, 1, route);
    }
    await flushPromises();

    expect(admissions).toBe(40);
    expect(mockSse).not.toHaveBeenCalled();
    expect(store.pending).toHaveLength(40);
  });

  it("holds at most four streams across separate Generate submissions", async () => {
    const store = useGenerationStore();
    const first = store.submitBatch({ ...req, seed: 200 }, 1);
    const second = store.submitBatch({ ...req, seed: 201 }, 1);
    const third = store.submitBatch({ ...req, seed: 202 }, 1);
    const fourth = store.submitBatch({ ...req, seed: 203 }, 1);
    const fifth = store.submitBatch({ ...req, seed: 204 }, 1);
    await flushPromises();

    expect(streams.map((stream) => stream.seed).sort()).toEqual([200, 201, 202, 203]);

    resolveStream(200);
    await flushPromises();
    expect(streams.map((stream) => stream.seed).sort()).toEqual([201, 202, 203, 204]);

    resolveStream(201);
    resolveStream(202);
    resolveStream(203);
    resolveStream(204);
    await Promise.all([
      first.settled,
      second.settled,
      third.settled,
      fourth.settled,
      fifth.settled,
    ]);
  });

  it("shares the four-stream host cap across overlapping batches", async () => {
    const store = useGenerationStore();
    const first = store.submitBatch({ ...req, seed: 400 }, 3);
    const second = store.submitBatch({ ...req, seed: 500 }, 3);
    await flushPromises();

    expect(streams).toHaveLength(4);

    resolveStream(400);
    await flushPromises();
    expect(streams).toHaveLength(4);
    resolveStream(401);
    await flushPromises();
    expect(streams).toHaveLength(4);

    while (streams.length > 0) {
      streams[0]!.resolve();
      await flushPromises();
    }
    await Promise.all([first.settled, second.settled]);
  });

  it("drains the mobile media backlog per target and keeps waiting jobs visible and cancellable", async () => {
    const store = useGenerationStore();
    const mobileRoute = (hostId: string, baseUrl: string) => ({
      hostId,
      label: hostId,
      kind: "remote" as const,
      target: { baseUrl, apiKey: `${hostId}-secret` },
      instanceId: `${hostId}-instance`,
      heterogeneousBatch: true,
      durableBatchOutcomes: true,
      mirrorRemoteOutput: false,
      retainEncodedResult: false,
      metadataOnlyCompletion: true,
    });
    const first = store.submitBatch(
      { ...req, seed: 600, source_image: "session-only-a" },
      5,
      mobileRoute("phone-a", "http://phone-a:7680"),
    );
    const second = store.submitBatch(
      { ...req, seed: 700, source_image: "session-only-b" },
      5,
      mobileRoute("phone-b", "http://phone-b:7680"),
    );

    await flushPromises();
    expect(store.jobs.map((job) => job.error)).toEqual(Array(10).fill(null));
    expect(store.pending).toHaveLength(10);
    expect(streams.filter((stream) => stream.target === "http://phone-a:7680")).toHaveLength(4);
    expect(streams.filter((stream) => stream.target === "http://phone-b:7680")).toHaveLength(4);

    expect(await store.cancel(first.jobs[4]!.clientId)).toBe(true);
    completeStream(600);
    await flushPromises();
    expect(streams.some((stream) => stream.seed === 604)).toBe(false);

    completeStream(700);
    await flushPromises();
    expect(streams.some((stream) => stream.seed === 704)).toBe(true);

    while (streams.length > 0) {
      completeStream(streams[0]!.seed);
      await flushPromises();
    }
    await Promise.all([first.settled, second.settled]);
  });

  it("releases a slot on a terminal frame even when the peer does not close", async () => {
    const store = useGenerationStore();
    const first = store.submitBatch({ ...req, seed: 300 }, 1);
    const second = store.submitBatch({ ...req, seed: 301 }, 1);
    const third = store.submitBatch({ ...req, seed: 302 }, 1);
    const fourth = store.submitBatch({ ...req, seed: 303 }, 1);
    const fifth = store.submitBatch({ ...req, seed: 304 }, 1);
    await flushPromises();
    expect(streams.map((stream) => stream.seed).sort()).toEqual([300, 301, 302, 303]);

    streams
      .find((stream) => stream.seed === 300)!
      .onEvent(
        "complete",
        JSON.stringify({
          image: btoa("generated"),
          format: "png",
          width: 1024,
          height: 1024,
          seed_used: 300,
          generation_time_ms: 100,
          model: req.model,
        }),
      );
    await flushPromises();

    expect(streams.map((stream) => stream.seed).sort()).toEqual([301, 302, 303, 304]);
    resolveStream(301);
    resolveStream(302);
    resolveStream(303);
    resolveStream(304);
    await Promise.all([
      first.settled,
      second.settled,
      third.settled,
      fourth.settled,
      fifth.settled,
    ]);
  });

  it("never opens a stream for a sibling cancelled before its turn", async () => {
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(req, 5);
    await flushPromises();

    // Cancel the last sibling while the first four hold the pool.
    jobs[4]!.status = "error";
    jobs[4]!.error = "Cancelled";

    resolveStream(100);
    await flushPromises();
    resolveStream(101);
    await flushPromises();
    resolveStream(102);
    await flushPromises();
    resolveStream(103);
    await flushPromises();
    await settled;

    const openedSeeds = mockSse.mock.calls.map((c) => (c[1].body as { seed: number }).seed);
    expect(openedSeeds).toContain(102);
    expect(openedSeeds).not.toContain(104);
  });
});
