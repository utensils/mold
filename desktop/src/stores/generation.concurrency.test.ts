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
  streamableMediaUrl: vi.fn().mockResolvedValue("blob:durable-result"),
  fetchGalleryMediaBytes: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
  saveOutputBytes: vi.fn().mockResolvedValue("saved.png"),
}));
const queueApi = vi.hoisted(() => ({ retryQueueJobRecoveringAmbiguity: vi.fn() }));
vi.mock("@studio/api/queuePlan", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/queuePlan")>()),
  retryQueueJobRecoveringAmbiguity: (...args: unknown[]) =>
    queueApi.retryQueueJobRecoveringAmbiguity(...args),
}));
vi.mock("../lib/notify", () => ({
  notifyGenerated: effectMocks.notifyGenerated,
  notifyGenerationFailed: effectMocks.notifyGenerationFailed,
}));
vi.mock("../lib/gallery/media", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/gallery/media")>()),
  streamableMediaUrl: effectMocks.streamableMediaUrl,
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
import { ApiError, apiFetchTo, apiJsonTo } from "../lib/api/client";
import { runWithConcurrency, useGenerationStore } from "./generation";
import { useHostsStore } from "./hosts";
import { useToastStore } from "./toasts";
import type { GenerateRequest } from "../lib/api/types";
import { DURABLE_GENERATION_STORAGE_KEY } from "../lib/durableGeneration";
import {
  createGenerationBatchTracker,
  reduceGenerationLifecycle,
} from "@studio/lib/generationLifecycle";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

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

  const req: GenerateRequest = {
    prompt: "a lighthouse",
    model: "flux-schnell:q8",
    width: 1024,
    height: 1024,
    steps: 4,
    seed: 100,
  };

  // The held-stream pool serves SEQUENCES alone now: a print is admitted
  // through the durable queue and never opens a stream, so every cap below is
  // exercised with auto-chained clips.
  const chainDecision = {
    kind: "chain" as const,
    clipFrames: 97,
    motionTail: 17,
    stageCount: 3,
  };
  const chainReq: GenerateRequest = { ...req, model: "ltx-2-19b-distilled:fp8", frames: 241 };

  function completeChainStream(seed: number) {
    const stream = streams.find((candidate) => candidate.seed === seed);
    stream!.onEvent(
      "complete",
      JSON.stringify({
        video: btoa("clip"),
        format: "mp4",
        width: 1024,
        height: 1024,
        frames: 241,
        fps: 24,
        generation_time_ms: 100,
        metadata: { seed, model: chainReq.model },
      }),
    );
  }

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
    queueApi.retryQueueJobRecoveringAmbiguity.mockReset();
    queueApi.retryQueueJobRecoveringAmbiguity.mockResolvedValue({ kind: "accepted" });
    effectMocks.notifyGenerated.mockClear();
    effectMocks.notifyGenerationFailed.mockClear();
    effectMocks.streamableMediaUrl.mockClear();
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

  it.each(["QuotaExceededError", "SecurityError"])(
    "admits once through the durable endpoint when recovery storage raises %s",
    async (name) => {
      const storage = new Map<string, string>();
      vi.stubGlobal("localStorage", {
        getItem: (key: string) => storage.get(key) ?? null,
        setItem: (key: string, value: string) => {
          if (key === DURABLE_GENERATION_STORAGE_KEY) {
            throw Object.assign(new Error("storage unavailable"), { name });
          }
          storage.set(key, value);
        },
      });
      // The unavailable-storage warning is deliberately once per module
      // session, so this case runs first in the file — a later test that has
      // already tripped it would leave nothing to observe.
      const store = useGenerationStore();
      durableApi.admit.mockImplementation(async (_target, body) => {
        const clientBatchId = (body as { client_batch_id: string }).client_batch_id;
        return {
          id: "storage-failure-batch",
          client_batch_id: clientBatchId,
          instance_id: "instance-1",
          durable: true,
          children: [
            {
              index: 1,
              job_id: "storage-failure-job",
              state: "queued",
              created_at_ms: 1,
              updated_at_ms: 1,
            },
          ],
        } as never;
      });

      const submitted = store.submitBatch(req, 1, {
        hostId: "hal9000",
        label: "hal9000",
        kind: "remote",
        target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
        instanceId: "instance-1",
        heterogeneousBatchMaxOutputs: 64,
        mirrorRemoteOutput: false,
      });
      await flushPromises();

      expect(durableApi.admit).toHaveBeenCalledTimes(1);
      expect(mockSse).not.toHaveBeenCalled();
      expect(submitted.jobs[0]).toMatchObject({
        id: "storage-failure-job",
        status: "queued",
        error: null,
      });
      expect(
        useToastStore().items.some(
          (toast) =>
            toast.kind === "warning" && toast.message.includes("Recovery storage is unavailable"),
        ),
      ).toBe(true);
    },
  );

  it("holds at most four sequence clips streaming at once", async () => {
    const store = useGenerationStore();
    const submitted = store.submitBatch(chainReq, 5, null, chainDecision);
    await flushPromises();
    expect(streams.map((s) => s.seed).sort()).toEqual([100, 101, 102, 103]);

    resolveStream(100);
    await flushPromises();
    expect(streams.map((s) => s.seed).sort()).toEqual([101, 102, 103, 104]);

    // Drain before leaving: a batch still holding streams would settle inside
    // the next test and pollute its module-scoped state.
    while (streams.length > 0) {
      streams[0]!.resolve();
      await flushPromises();
    }
    await submitted.settled;
  });
  it("refuses held Retry after the owning server instance is replaced", async () => {
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
    durableApi.admit.mockImplementation(async (_target, body) => ({
      id: "held-batch",
      client_batch_id: (body as { client_batch_id: string }).client_batch_id,
      instance_id: "instance-1",
      durable: true,
      children: [
        {
          index: 1,
          job_id: "held-job",
          state: "held",
          error: "model preparation failed",
          retryable: true,
          created_at_ms: 1,
          updated_at_ms: 2,
        },
      ],
    }));

    const submitted = store.submitBatch(req, 1, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatchMaxOutputs: 64,
    });
    await flushPromises();
    expect(submitted.jobs[0]).toMatchObject({ retryable: true, id: "held-job" });

    hosts.extras[0]!.instanceId = "replacement-instance";
    await store.reconcileDurableHost("hal9000");
    expect(submitted.jobs[0]).toMatchObject({
      retryable: false,
      interrupted: true,
      stage: "Original machine identity changed — outcome unknown",
    });
    await expect(store.retryHeld(submitted.jobs[0]!.clientId)).rejects.toThrow("not retryable");
    expect(queueApi.retryQueueJobRecoveringAmbiguity).not.toHaveBeenCalled();
  });

  it("retries a held child with its complete durable admission authority", async () => {
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
    durableApi.admit.mockImplementation(async (_target, body) => ({
      id: "held-batch",
      client_batch_id: (body as { client_batch_id: string }).client_batch_id,
      instance_id: "instance-1",
      durable: true,
      children: [
        {
          index: 1,
          job_id: "held/job",
          state: "held",
          error: "model preparation failed",
          retryable: true,
          created_at_ms: 1,
          updated_at_ms: 2,
        },
      ],
    }));

    const submitted = store.submitBatch(req, 1, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatchMaxOutputs: 64,
    });
    await flushPromises();

    const confirmation = deferred<{ kind: "accepted" }>();
    queueApi.retryQueueJobRecoveringAmbiguity.mockReturnValue(confirmation.promise);
    const retry = store.retryHeld(submitted.jobs[0]!.clientId);

    expect(submitted.jobs[0]).toMatchObject({
      retryable: false,
      retrying: true,
    });
    confirmation.resolve({ kind: "accepted" });
    await retry;

    expect(queueApi.retryQueueJobRecoveringAmbiguity).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      {
        instanceId: "instance-1",
        batchId: "held-batch",
        clientBatchId: expect.any(String),
        jobId: "held/job",
      },
    );
  });

  it("admits singleton and chunks Batch N without held generation streams", async () => {
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
      heterogeneousBatchMaxOutputs: 2,
      mirrorRemoteOutput: false,
    };

    const singleton = store.submitBatch(req, 1, route);
    const batch = store.submitBatch(req, 5, route);
    await flushPromises();

    expect(singleton.jobs.map((job) => job.id)).toEqual(["job-1-1"]);
    expect(batch.jobs.map((job) => job.id)).toEqual([
      "job-2-1",
      "job-2-2",
      "job-3-1",
      "job-3-2",
      "job-4-1",
    ]);
    expect(durableApi.admit.mock.calls.map((call) => call[1].requests.length)).toEqual([
      1, 2, 2, 1,
    ]);
    expect(mockSse).not.toHaveBeenCalled();
  });

  it.each([
    ["commit-then-500", new ApiError("response lost", 500)],
    ["disconnect", new TypeError("response lost")],
  ])(
    "recovers an ambiguous durable %s POST by client id without fallback",
    async (_case, failure) => {
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
        throw failure;
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
        heterogeneousBatchMaxOutputs: 64,
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
    },
  );

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
      heterogeneousBatchMaxOutputs: 64,
      mirrorRemoteOutput: true,
    });
    await flushPromises();

    store.onDurableEvent("hal9000", "authority", '{"instance_id":"instance-1"}');
    await flushPromises();
    expect(submitted.jobs[0]!.status).toBe("loading");

    store.onDurableEvent("hal9000", "event", '{"type":"job_ended","id":"job-1"}');
    await flushPromises();
    expect(submitted.jobs[0]!.status).toBe("loading");

    phase = "complete";
    store.onDurableEvent(
      "hal9000",
      "event",
      '{"type":"job_state_committed","id":"committed-before-client-map"}',
    );
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

  it("reconciles mapped lifecycle hints only against their owning durable batch", async () => {
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
    const admitted = new Map<string, { batchId: string; jobId: string }>();
    let ordinal = 0;
    durableApi.admit.mockImplementation(async (_target, body) => {
      const clientBatchId = (body as { client_batch_id: string }).client_batch_id;
      ordinal += 1;
      const identity = { batchId: `batch-scope-${ordinal}`, jobId: `job-scope-${ordinal}` };
      admitted.set(clientBatchId, identity);
      return {
        id: identity.batchId,
        client_batch_id: clientBatchId,
        instance_id: "instance-1",
        durable: true,
        children: [
          {
            index: 1,
            job_id: identity.jobId,
            state: "queued",
            created_at_ms: 1,
            updated_at_ms: 1,
          },
        ],
      };
    });
    durableApi.reconcile.mockImplementation(async (_target, body) => {
      const requested = new Set((body as { batch_ids?: string[] }).batch_ids ?? []);
      return {
        instance_id: "instance-1",
        batches: [...admitted.entries()]
          .filter(([, identity]) => requested.has(identity.batchId))
          .map(([clientBatchId, identity]) => ({
            id: identity.batchId,
            client_batch_id: clientBatchId,
            instance_id: "instance-1",
            durable: true,
            children: [
              {
                index: 1,
                job_id: identity.jobId,
                state: "running",
                created_at_ms: 1,
                updated_at_ms: 2,
              },
            ],
          })),
        missing: { client_batch_ids: [], batch_ids: [] },
      };
    });
    const route = {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote" as const,
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatchMaxOutputs: 64,
      mirrorRemoteOutput: false,
    };
    store.submitBatch(req, 1, route);
    store.submitBatch(req, 1, route);
    await flushPromises();
    expect(admitted.size).toBe(2);
    durableApi.reconcile.mockClear();

    store.onDurableEvent(
      "hal9000",
      "event",
      JSON.stringify({ type: "job_ended", id: "job-scope-1" }),
    );
    await flushPromises();

    expect(durableApi.reconcile).toHaveBeenCalledTimes(1);
    expect(durableApi.reconcile.mock.calls[0]![1]).toMatchObject({
      batch_ids: ["batch-scope-1"],
    });
  });

  it("guarantees a follow-up reconcile when an invalidation arrives in flight", async () => {
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
    const firstRead = deferred<Record<string, unknown>>();
    const batch = (state: "queued" | "running" | "complete") => ({
      id: "batch-dirty",
      client_batch_id: clientBatchId,
      instance_id: "instance-1",
      durable: true,
      children: [
        {
          index: 1,
          job_id: "job-dirty",
          state,
          created_at_ms: 1,
          updated_at_ms: state === "queued" ? 1 : state === "running" ? 2 : 3,
          ...(state === "complete"
            ? { completed_at_ms: 3, result: { filename: "dirty-finished.png" } }
            : {}),
        },
      ],
    });
    durableApi.admit.mockImplementation(async (_target, body) => {
      clientBatchId = (body as { client_batch_id: string }).client_batch_id;
      return batch("queued");
    });
    durableApi.reconcile
      .mockImplementationOnce(() => firstRead.promise)
      .mockImplementationOnce(async () => ({
        instance_id: "instance-1",
        batches: [batch("complete")],
        missing: { client_batch_ids: [], batch_ids: [] },
      }));

    const submitted = store.submitBatch(req, 1, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatchMaxOutputs: 64,
      mirrorRemoteOutput: false,
    });
    await flushPromises();
    store.onDurableEvent("hal9000", "authority", '{"instance_id":"instance-1"}');
    await flushPromises();
    expect(durableApi.reconcile).toHaveBeenCalledTimes(1);

    store.onDurableEvent("hal9000", "event", '{"type":"job_started","id":"job-dirty"}');
    firstRead.resolve({
      instance_id: "instance-1",
      batches: [batch("running")],
      missing: { client_batch_ids: [], batch_ids: [] },
    });

    await submitted.settled;
    expect(durableApi.reconcile).toHaveBeenCalledTimes(2);
    expect(submitted.jobs[0]).toMatchObject({
      status: "complete",
      result: { filename: "dirty-finished.png" },
    });
  });

  it("persists a pre-admission cancel and deletes the exact id after admission", async () => {
    const storage = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => void storage.set(key, value),
    });
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
    const admission = deferred<Record<string, unknown>>();
    let clientBatchId = "";
    const batch = (state: "queued" | "cancelled") => ({
      id: "batch-pre-id",
      client_batch_id: clientBatchId,
      instance_id: "instance-1",
      durable: true,
      children: [
        {
          index: 1,
          job_id: "server-job-pre-id",
          state,
          created_at_ms: 1,
          updated_at_ms: state === "queued" ? 2 : 3,
          ...(state === "cancelled" ? { completed_at_ms: 3 } : {}),
        },
      ],
    });
    durableApi.admit.mockImplementation((_target, body) => {
      clientBatchId = (body as { client_batch_id: string }).client_batch_id;
      return admission.promise;
    });
    durableApi.reconcile.mockImplementation(async () => ({
      instance_id: "instance-1",
      batches: [batch("cancelled")],
      missing: { client_batch_ids: [], batch_ids: [] },
    }));
    const submitted = store.submitBatch(req, 1, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatchMaxOutputs: 64,
      mirrorRemoteOutput: false,
    });
    await flushPromises();

    const cancelled = store.cancel(submitted.jobs[0]!.clientId);
    await flushPromises();
    expect(JSON.parse(storage.get(DURABLE_GENERATION_STORAGE_KEY) ?? "null")).toMatchObject({
      records: [{ cancelRequestedChildIndexes: [1] }],
    });
    await expect(cancelled).resolves.toBe(false);
    expect(durableApi.reconcile).not.toHaveBeenCalled();
    expect(apiFetchTo).not.toHaveBeenCalled();

    admission.resolve(batch("queued"));
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      "/api/queue/server-job-pre-id",
      { method: "DELETE" },
    );
    expect(submitted.jobs[0]).toMatchObject({ status: "error", error: "Cancelled" });
  });

  it("sends durable DELETE before terminal reconciliation", async () => {
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
    const calls: string[] = [];
    let clientBatchId = "";
    const batch = (state: "queued" | "cancelled") => ({
      id: "batch-immediate",
      client_batch_id: clientBatchId,
      instance_id: "instance-1",
      durable: true as const,
      children: [
        {
          index: 1,
          job_id: "job-immediate",
          state,
          created_at_ms: 1,
          updated_at_ms: state === "queued" ? 1 : 2,
          ...(state === "cancelled" ? { completed_at_ms: 2 } : {}),
        },
      ],
    });
    durableApi.admit.mockImplementation(async (_target, body) => {
      clientBatchId = (body as { client_batch_id: string }).client_batch_id;
      return batch("queued");
    });
    vi.mocked(apiFetchTo).mockImplementationOnce(async () => {
      calls.push("delete");
      return new Response(null, { status: 204 });
    });
    durableApi.reconcile.mockImplementationOnce(async () => {
      calls.push("reconcile");
      return {
        instance_id: "instance-1",
        batches: [batch("cancelled")],
        missing: { client_batch_ids: [], batch_ids: [] },
      };
    });

    const submitted = store.submitBatch(req, 1, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatchMaxOutputs: 64,
      mirrorRemoteOutput: false,
    });
    await submitted.admitted;

    await expect(store.cancel(submitted.jobs[0]!.clientId)).resolves.toBe(true);
    expect(calls).toEqual(["delete", "reconcile"]);
    expect(submitted.jobs[0]).toMatchObject({ status: "error", error: "Cancelled" });
  });

  it("restores cancelled durable rows until explicit removal", async () => {
    const storage = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => void storage.set(key, value),
    });
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
    durableApi.admit.mockImplementation(async (_target, body) => ({
      id: "cancelled-batch",
      client_batch_id: (body as { client_batch_id: string }).client_batch_id,
      instance_id: "instance-1",
      durable: true,
      children: [
        {
          index: 1,
          job_id: "cancelled-job",
          state: "cancelled",
          error: "Cancelled",
          created_at_ms: 1,
          updated_at_ms: 2,
          completed_at_ms: 2,
        },
      ],
    }));

    const store = useGenerationStore();
    const submitted = store.submitBatch(req, 1, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatchMaxOutputs: 64,
      mirrorRemoteOutput: false,
    });
    await submitted.settled;
    expect(submitted.jobs[0]).toMatchObject({ status: "error", error: "Cancelled" });
    expect(JSON.parse(storage.get(DURABLE_GENERATION_STORAGE_KEY) ?? "null").records).toHaveLength(
      1,
    );

    store.resetJobs();
    store.resumeDurableGenerations();
    expect(store.jobs).toHaveLength(1);
    expect(store.jobs[0]).toMatchObject({ status: "error", error: "Cancelled" });

    expect(store.removeSettled(store.jobs[0]!.clientId)).toBe(true);
    await flushPromises();
    expect(JSON.parse(storage.get(DURABLE_GENERATION_STORAGE_KEY) ?? "null").records).toEqual([]);

    store.resetJobs();
    store.resumeDurableGenerations();
    expect(store.jobs).toEqual([]);
  });

  it("keeps stable batch membership after one terminal child is removed", async () => {
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
    let clientBatchId = "";
    const durableBatch = (secondState: "queued" | "cancelled") => ({
      id: "mixed-batch",
      client_batch_id: clientBatchId,
      instance_id: "instance-1",
      durable: true as const,
      children: [
        {
          index: 1,
          job_id: "mixed-job-1",
          state: "cancelled",
          error: "Cancelled",
          created_at_ms: 1,
          updated_at_ms: 2,
          completed_at_ms: 2,
        },
        {
          index: 2,
          job_id: "mixed-job-2",
          state: secondState,
          ...(secondState === "cancelled" ? { error: "Cancelled", completed_at_ms: 3 } : {}),
          created_at_ms: 1,
          updated_at_ms: secondState === "queued" ? 1 : 3,
        },
      ],
    });
    durableApi.admit.mockImplementation(async (_target, body) => {
      clientBatchId = (body as { client_batch_id: string }).client_batch_id;
      return durableBatch("queued");
    });

    const store = useGenerationStore();
    const submitted = store.submitBatch(req, 2, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatchMaxOutputs: 64,
      mirrorRemoteOutput: false,
    });
    await submitted.admitted;
    expect(store.removeSettled(submitted.jobs[0]!.clientId)).toBe(true);
    expect(store.jobs).toEqual([submitted.jobs[1]]);

    durableApi.reconcile.mockResolvedValueOnce({
      instance_id: "instance-1",
      batches: [durableBatch("cancelled")],
      missing: { client_batch_ids: [], batch_ids: [] },
    });
    await store.reconcileDurableAll();

    const settledJobs = await submitted.settled;
    expect(settledJobs).toHaveLength(2);
    expect(settledJobs[0]).toBe(submitted.jobs[0]);
    expect(settledJobs[1]).toBe(submitted.jobs[1]);
  });

  it("restores dismissed children for pending batch effects without showing them", async () => {
    const storage = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => void storage.set(key, value),
    });
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
    let clientBatchId = "";
    const durableBatch = (secondState: "queued" | "cancelled") => ({
      id: "restart-batch",
      client_batch_id: clientBatchId,
      instance_id: "instance-1",
      durable: true as const,
      children: [
        {
          index: 1,
          job_id: "restart-job-1",
          state: "complete",
          result: { filename: "dismissed-complete.png" },
          created_at_ms: 1,
          updated_at_ms: 2,
          completed_at_ms: 2,
        },
        {
          index: 2,
          job_id: "restart-job-2",
          state: secondState,
          ...(secondState === "cancelled" ? { error: "Cancelled", completed_at_ms: 3 } : {}),
          created_at_ms: 1,
          updated_at_ms: secondState === "queued" ? 1 : 3,
        },
      ],
    });
    durableApi.admit.mockImplementation(async (_target, body) => {
      clientBatchId = (body as { client_batch_id: string }).client_batch_id;
      return durableBatch("queued");
    });

    const store = useGenerationStore();
    const submitted = store.submitBatch(req, 2, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatchMaxOutputs: 64,
      mirrorRemoteOutput: false,
    });
    await submitted.admitted;
    expect(store.removeSettled(submitted.jobs[0]!.clientId)).toBe(true);
    effectMocks.streamableMediaUrl.mockClear();

    durableApi.reconcile.mockResolvedValue({
      instance_id: "instance-1",
      batches: [durableBatch("cancelled")],
      missing: { client_batch_ids: [], batch_ids: [] },
    });
    store.resetJobs();
    store.resumeDurableGenerations();

    expect(store.jobs).toHaveLength(1);
    expect(store.jobs[0]?.id).toBe("restart-job-2");
    await vi.waitFor(() =>
      expect(effectMocks.notifyGenerated).toHaveBeenCalledWith(
        "Recovered generation",
        "dismissed-complete.png",
      ),
    );
    expect(effectMocks.streamableMediaUrl).not.toHaveBeenCalled();
    expect(store.jobs).toHaveLength(1);
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
            cancelRequestedChildIndexes: [],
            effectReceipts: [],
          },
          {
            tracker: staleTracker,
            hostLabel: "old hal9000",
            hostKind: "remote",
            mirrorRemoteOutput: false,
            children: [child],
            cancelRequestedChildIndexes: [],
            effectReceipts: [],
          },
          {
            tracker: terminalTracker,
            hostLabel: "old hal9000",
            hostKind: "remote",
            mirrorRemoteOutput: true,
            children: [child],
            cancelRequestedChildIndexes: [],
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

  it("suppresses restored completion per already-terminal child, not for its live sibling", () => {
    const storage = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => void storage.set(key, value),
    });
    let tracker = createGenerationBatchTracker({
      hostId: "hal9000",
      expectedInstanceId: "instance-1",
      clientBatchId: "client-partial",
      submittedAtMs: 1,
    });
    tracker = reduceGenerationLifecycle(tracker, {
      type: "batch_snapshot",
      batch: {
        id: "batch-partial",
        client_batch_id: "client-partial",
        instance_id: "instance-1",
        durable: true,
        children: [
          {
            index: 1,
            job_id: "job-finished-before-restart",
            state: "complete",
            created_at_ms: 1,
            updated_at_ms: 2,
            completed_at_ms: 2,
            result: { filename: "restored.png" },
          },
          {
            index: 2,
            job_id: "job-still-live",
            state: "queued",
            created_at_ms: 1,
            updated_at_ms: 2,
          },
        ],
      },
    });
    const child = (index: number) => ({
      index,
      clientId: null,
      model: req.model,
      width: req.width,
      height: req.height,
      steps: req.steps,
      guidance: 1,
      seed: (req.seed ?? 0) + index - 1,
      format: "png" as const,
    });
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
            children: [child(1), child(2)],
            cancelRequestedChildIndexes: [],
            effectReceipts: [],
          },
        ],
      }),
    );

    const store = useGenerationStore();
    store.resumeDurableGenerations();

    expect(store.jobs.map((job) => job.suppressFreshCompletion)).toEqual([true, false]);
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
      heterogeneousBatchMaxOutputs: 64,
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
      heterogeneousBatchMaxOutputs: 64,
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

  it("admits supported media through the encrypted durable capability without a stream slot", async () => {
    const storage = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => void storage.set(key, value),
    });
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
    durableApi.admit.mockImplementation(async (_target, body) => {
      const client = (body as { client_batch_id: string }).client_batch_id;
      return {
        id: "media-batch",
        client_batch_id: client,
        instance_id: "instance-1",
        durable: true,
        children: [
          {
            index: 1,
            job_id: "media-job",
            state: "queued",
            created_at_ms: 1,
            updated_at_ms: 1,
          },
        ],
      };
    });

    store.submitBatch({ ...req, source_image: "PRIVATE-DURABLE-SOURCE" }, 1, {
      hostId: "hal9000",
      label: "hal9000",
      kind: "remote",
      target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
      instanceId: "instance-1",
      heterogeneousBatchMaxOutputs: 64,
      durableMedia: {
        protocol_version: 2,
        encrypted_at_rest: true,
        generate_request_media: true,
        identity: true,
        h3_references: false,
        private_h3: true,
      },
      mirrorRemoteOutput: false,
    });
    await flushPromises();

    expect(durableApi.admit).toHaveBeenCalledTimes(1);
    expect(durableApi.admit.mock.calls[0]![1].requests[0]).toMatchObject({
      source_image: "PRIVATE-DURABLE-SOURCE",
    });
    expect(mockSse).not.toHaveBeenCalled();
    expect(storage.get(DURABLE_GENERATION_STORAGE_KEY) ?? "").not.toContain(
      "PRIVATE-DURABLE-SOURCE",
    );
  });

  it("refuses an opaque H3 family the machine has no private contract for", async () => {
    const store = useGenerationStore();
    expect(() =>
      store.submitBatch(
        {
          ...req,
          model: "hf:opaque-h3-checkpoint",
          source_image: "PRIVATE-H3-SOURCE",
        },
        1,
        {
          hostId: "hal9000",
          label: "hal9000",
          kind: "remote",
          target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
          instanceId: "instance-1",
          heterogeneousBatchMaxOutputs: 64,
          durableMedia: {
            protocol_version: 2,
            encrypted_at_rest: true,
            generate_request_media: true,
            identity: true,
            h3_references: false,
            private_h3: false,
          },
          modelFamily: "minimax-h3",
        },
      ),
    ).toThrow("cannot store MiniMax H3 request media durably");
    await flushPromises();

    expect(durableApi.admit).not.toHaveBeenCalled();
    expect(mockSse).not.toHaveBeenCalled();
    expect(store.jobs).toHaveLength(0);
  });
  it("admits canonical v2 H3 through the durable batch transport", async () => {
    durableApi.admit.mockImplementation(() => new Promise(() => {}));
    const store = useGenerationStore();
    store.submitBatch(
      {
        ...req,
        model: "hf:opaque-h3-checkpoint",
        source_image: "PRIVATE-H3-SOURCE",
      },
      1,
      {
        hostId: "hal9000",
        label: "hal9000",
        kind: "remote",
        target: { baseUrl: "http://hal9000:7680", apiKey: "fresh-key" },
        instanceId: "instance-1",
        heterogeneousBatchMaxOutputs: 64,
        durableMedia: {
          protocol_version: 3,
          encrypted_at_rest: true,
          generate_request_media: true,
          identity: true,
          h3_references: true,
          private_h3: true,
        },
        modelFamily: "minimax-h3",
      },
    );
    await flushPromises();

    expect(durableApi.admit).toHaveBeenCalledTimes(1);
    expect(durableApi.admit.mock.calls[0]![1].requests[0]).toMatchObject({
      model: "hf:opaque-h3-checkpoint",
      source_image: "PRIVATE-H3-SOURCE",
    });
    expect(mockSse).not.toHaveBeenCalled();
  });

  it("holds at most four sequence streams across separate Generate submissions", async () => {
    const store = useGenerationStore();
    const first = store.submitBatch({ ...chainReq, seed: 200 }, 1, null, chainDecision);
    const second = store.submitBatch({ ...chainReq, seed: 201 }, 1, null, chainDecision);
    const third = store.submitBatch({ ...chainReq, seed: 202 }, 1, null, chainDecision);
    const fourth = store.submitBatch({ ...chainReq, seed: 203 }, 1, null, chainDecision);
    const fifth = store.submitBatch({ ...chainReq, seed: 204 }, 1, null, chainDecision);
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
  it("shares the four-stream host cap across overlapping sequence batches", async () => {
    const store = useGenerationStore();
    const first = store.submitBatch({ ...chainReq, seed: 400 }, 3, null, chainDecision);
    const second = store.submitBatch({ ...chainReq, seed: 500 }, 3, null, chainDecision);
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
  it("drains the sequence backlog per target and keeps waiting clips visible and cancellable", async () => {
    const store = useGenerationStore();
    const mobileRoute = (hostId: string, baseUrl: string) => ({
      hostId,
      label: hostId,
      kind: "remote" as const,
      target: { baseUrl, apiKey: `${hostId}-secret` },
      instanceId: `${hostId}-instance`,
      heterogeneousBatchMaxOutputs: 64,
      mirrorRemoteOutput: false,
      retainEncodedResult: false,
      metadataOnlyCompletion: true,
    });
    const first = store.submitBatch(
      { ...chainReq, seed: 600 },
      5,
      mobileRoute("phone-a", "http://phone-a:7680"),
      chainDecision,
    );
    const second = store.submitBatch(
      { ...chainReq, seed: 700 },
      5,
      mobileRoute("phone-b", "http://phone-b:7680"),
      chainDecision,
    );

    await flushPromises();
    expect(store.jobs.map((job) => job.error)).toEqual(Array(10).fill(null));
    expect(store.pending).toHaveLength(10);
    expect(streams.filter((stream) => stream.target === "http://phone-a:7680")).toHaveLength(4);
    expect(streams.filter((stream) => stream.target === "http://phone-b:7680")).toHaveLength(4);

    expect(await store.cancel(first.jobs[4]!.clientId)).toBe(true);
    completeChainStream(600);
    await flushPromises();
    expect(streams.some((stream) => stream.seed === 604)).toBe(false);

    completeChainStream(700);
    await flushPromises();
    expect(streams.some((stream) => stream.seed === 704)).toBe(true);

    while (streams.length > 0) {
      completeChainStream(streams[0]!.seed);
      await flushPromises();
    }
    await Promise.all([first.settled, second.settled]);
  });
  it("releases a slot on a terminal sequence frame even when the peer does not close", async () => {
    const store = useGenerationStore();
    const first = store.submitBatch({ ...chainReq, seed: 300 }, 1, null, chainDecision);
    const second = store.submitBatch({ ...chainReq, seed: 301 }, 1, null, chainDecision);
    const third = store.submitBatch({ ...chainReq, seed: 302 }, 1, null, chainDecision);
    const fourth = store.submitBatch({ ...chainReq, seed: 303 }, 1, null, chainDecision);
    const fifth = store.submitBatch({ ...chainReq, seed: 304 }, 1, null, chainDecision);
    await flushPromises();
    expect(streams.map((stream) => stream.seed).sort()).toEqual([300, 301, 302, 303]);

    completeChainStream(300);
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
  it("never opens a stream for a sequence sibling cancelled before its turn", async () => {
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(chainReq, 5, null, chainDecision);
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
