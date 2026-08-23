import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GenerateRequestWire, GalleryImage } from "../types";
import type { HostRoute } from "../lib/hostRouting";
import { GENERATION_REQUEST_MEDIA_FIELDS } from "../lib/generationRequestMedia";
import type {
  GenerationBatchStatus,
  GenerationBatchStatusResponse,
} from "@studio/api/generationAdmission";

const admitGenerationBatch = vi.hoisted(() => vi.fn());
const lookupGenerationBatchByClientId = vi.hoisted(() => vi.fn());
const reconcileGenerationBatches = vi.hoisted(() => vi.fn());
const fetchEventSource = vi.hoisted(() => vi.fn(() => new Promise(() => {})));
const generateStream = vi.hoisted(() => vi.fn().mockResolvedValue(undefined));
const cancelQueueJob = vi.hoisted(() => vi.fn().mockResolvedValue(undefined));
const listGalleryFrom = vi.hoisted(() => vi.fn());
const fetchGalleryBlob = vi.hoisted(() => vi.fn());

vi.mock("@studio/api/generationAdmission", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/generationAdmission")>()),
  admitGenerationBatch,
  lookupGenerationBatchByClientId,
  reconcileGenerationBatches,
}));

vi.mock("@microsoft/fetch-event-source", () => ({ fetchEventSource }));

vi.mock("../api", () => ({
  cancelQueueJob,
  fetchQueue: vi.fn().mockResolvedValue({ entries: [] }),
  generateStream,
  generateChainStream: vi.fn().mockResolvedValue(undefined),
  listGalleryFrom,
}));

vi.mock("../lib/galleryMedia", () => ({ fetchGalleryBlob }));

import { __testing__, useGenerateStream, type Job } from "./useGenerateStream";

const nativeStorageSetItem = localStorage.setItem;

function request(prompt = "a patient red fox"): GenerateRequestWire {
  return {
    prompt,
    model: "flux-dev:fp8",
    width: 512,
    height: 512,
    steps: 8,
    guidance: 3,
    batch_size: 1,
    output_format: "png",
  };
}

const route: HostRoute = {
  hostId: "render-box",
  label: "Render box",
  target: { baseUrl: "http://render-box:7680", apiKey: "secret" },
  instanceId: "instance-1",
  durableGeneration: {
    heterogeneous_batch: true,
    durable_batch_outcomes: true,
  },
  eventsAvailable: true,
};

function batch(
  clientBatchId: string,
  states: Array<
    "accepted" | "queued" | "running" | "complete" | "failed" | "cancelled"
  > = ["queued"],
  overrides: Partial<GenerationBatchStatus> = {},
): GenerationBatchStatus {
  return {
    id: `server-${clientBatchId}`,
    client_batch_id: clientBatchId,
    instance_id: "instance-1",
    durable: true,
    children: states.map((state, offset) => ({
      index: offset + 1,
      job_id: `job-${clientBatchId}-${offset + 1}`,
      state,
      created_at_ms: 10,
      updated_at_ms: 20 + offset,
      ...(state === "complete"
        ? {
            completed_at_ms: 30,
            result: { filename: `print-${offset + 1}.png` },
          }
        : {}),
    })),
    ...overrides,
  };
}

function statusResponse(
  batches: GenerationBatchStatus[],
): GenerationBatchStatusResponse {
  return {
    instance_id: "instance-1",
    batches,
    missing: { client_batch_ids: [], batch_ids: [] },
  };
}

function gallery(filename: string): GalleryImage {
  return {
    filename,
    timestamp: 1,
    format: "png",
    metadata: {
      prompt: "a patient red fox",
      model: "flux-dev:fp8",
      seed: 42,
      steps: 8,
      guidance: 3,
      width: 512,
      height: 512,
      version: "test",
    },
  };
}

function clearJobs(): void {
  const stream = useGenerateStream();
  for (const job of [...stream.jobs.value]) stream.remove(job.id);
}

beforeEach(() => {
  clearJobs();
  __testing__.resetDurableLifecycleForTests();
  localStorage.clear();
  admitGenerationBatch.mockReset();
  lookupGenerationBatchByClientId.mockReset();
  reconcileGenerationBatches.mockReset();
  fetchEventSource.mockClear();
  generateStream.mockClear();
  cancelQueueJob.mockClear();
  listGalleryFrom.mockReset();
  fetchGalleryBlob.mockReset();
  listGalleryFrom.mockResolvedValue([
    gallery("print-1.png"),
    gallery("print-2.png"),
  ]);
  fetchGalleryBlob.mockResolvedValue(new Blob(["media"]));
  lookupGenerationBatchByClientId.mockResolvedValue({ kind: "missing" });
});

describe("web durable generation lifecycle", () => {
  it("has no browser submission-count cap and releases every POST immediately", () => {
    admitGenerationBatch.mockImplementation(() => new Promise(() => {}));
    const stream = useGenerateStream();

    const ids = Array.from({ length: 9 }, (_, index) =>
      stream.submit(request(`print ${index}`), { kind: "single" }, route),
    );

    expect(new Set(ids).size).toBe(9);
    expect(admitGenerationBatch).toHaveBeenCalledTimes(9);
    expect(generateStream).not.toHaveBeenCalled();
    expect(
      stream.jobs.value.filter((job) => job.state === "running"),
    ).toHaveLength(9);
  });

  it("maps singleton and Batch N children from one immediate durable admission", async () => {
    admitGenerationBatch.mockImplementation(
      (
        _target: unknown,
        body: { client_batch_id: string; requests: unknown[] },
      ) =>
        Promise.resolve(
          batch(
            body.client_batch_id,
            body.requests.map(() => "queued"),
          ),
        ),
    );
    const stream = useGenerateStream();

    const singleton = stream.submit(request("one"), { kind: "single" }, route);
    const siblings = stream.submitBatch(
      [request("two"), request("three"), request("four")],
      { kind: "single" },
      route,
    );

    await vi.waitFor(() =>
      expect(
        [singleton, ...siblings].map(
          (id) => stream.jobs.value.find((job) => job.id === id)?.serverId,
        ),
      ).toEqual([
        expect.stringMatching(/^job-/),
        expect.stringMatching(/^job-/),
        expect.stringMatching(/^job-/),
        expect.stringMatching(/^job-/),
      ]),
    );
    expect(admitGenerationBatch.mock.calls[1]![1].requests).toHaveLength(3);
  });

  it("fails closed before admission when the durable recovery fence cannot persist", async () => {
    const storage = vi
      .spyOn(localStorage, "setItem")
      .mockImplementation(function (key, value) {
        if (key === "mold.generate.jobs") {
          throw new DOMException("quota exceeded", "QuotaExceededError");
        }
        return Reflect.apply(nativeStorageSetItem, localStorage, [key, value]);
      });
    try {
      const stream = useGenerateStream();

      const id = stream.submit(request(), { kind: "single" }, route);
      await Promise.resolve();

      const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
      expect(job.state).toBe("error");
      expect(job.error).toMatch(/Nothing was submitted/);
      expect(job.durableBatch).toBeUndefined();
      expect(admitGenerationBatch).not.toHaveBeenCalled();
      expect(generateStream).not.toHaveBeenCalled();
      expect(lookupGenerationBatchByClientId).not.toHaveBeenCalled();
      expect(fetchEventSource).not.toHaveBeenCalled();
    } finally {
      storage.mockRestore();
    }
  });

  it("recovers an ambiguous POST only by UUID and never falls back after dispatch", async () => {
    let generationWrites = 0;
    const storage = vi
      .spyOn(localStorage, "setItem")
      .mockImplementation(function (key, value) {
        if (key === "mold.generate.jobs" && generationWrites++ > 0) {
          throw new DOMException("quota exceeded", "QuotaExceededError");
        }
        return Reflect.apply(nativeStorageSetItem, localStorage, [key, value]);
      });
    try {
      admitGenerationBatch.mockRejectedValue(
        new TypeError("connection closed"),
      );
      lookupGenerationBatchByClientId.mockImplementation(
        (_target: unknown, clientBatchId: string) =>
          Promise.resolve({ kind: "found", batch: batch(clientBatchId) }),
      );
      const stream = useGenerateStream();
      const id = stream.submit(request(), { kind: "single" }, route);

      await vi.waitFor(() =>
        expect(
          stream.jobs.value.find((job) => job.id === id)?.serverId,
        ).toMatch(/^job-/),
      );
      const persisted = localStorage.getItem("mold.generate.jobs")!;
      const clientBatchId = stream.jobs.value.find((job) => job.id === id)!
        .durableBatch!.clientBatchId;
      expect(persisted).toContain(clientBatchId);
      expect(persisted).not.toContain("secret");
      expect(lookupGenerationBatchByClientId).toHaveBeenCalledWith(
        expect.anything(),
        clientBatchId,
      );
      expect(generateStream).not.toHaveBeenCalled();
    } finally {
      storage.mockRestore();
    }
  });

  it("keeps recovery-record media redaction as defense in depth", () => {
    const redacted = __testing__.durablePersistenceSafeRequest({
      ...request(),
      source_image: "source",
      edit_images: ["edit"],
      references: [],
      id_image: "identity",
      mask_image: "mask",
      control_image: "control",
      audio_file: "audio",
      audio_file_path: "/audio",
      source_video: "source-video",
      source_video_path: "/source-video",
      extend_video: "extend-video",
      extend_video_path: "/extend-video",
      keyframes: [{ frame: 0, image: "keyframe" }],
    }) as unknown as Record<string, unknown>;

    for (const field of GENERATION_REQUEST_MEDIA_FIELDS) {
      expect(redacted).not.toHaveProperty(field);
    }
  });

  it("bulk-reconciles on event gaps and reconnect opens", async () => {
    admitGenerationBatch.mockImplementation(
      (_target: unknown, body: { client_batch_id: string }) =>
        Promise.resolve(batch(body.client_batch_id)),
    );
    reconcileGenerationBatches.mockImplementation(() => {
      const tracked = useGenerateStream().jobs.value.find(
        (job) => job.durableBatch,
      );
      return Promise.resolve(
        statusResponse([batch(tracked!.durableBatch!.clientBatchId)]),
      );
    });
    const stream = useGenerateStream();
    stream.submit(request(), { kind: "single" }, route);
    await vi.waitFor(() => expect(fetchEventSource).toHaveBeenCalled());
    const options = (
      fetchEventSource.mock.calls as unknown as Array<
        [
          string,
          {
            onopen: (response: Response) => Promise<void>;
            onmessage: (message: { event: string; data: string }) => void;
          },
        ]
      >
    )[0]![1];

    await options.onopen(new Response(null, { status: 200 }));
    options.onmessage({
      event: "resync_required",
      data: JSON.stringify({ instance_id: "instance-1", missed_events: 3 }),
    });

    await vi.waitFor(() =>
      expect(reconcileGenerationBatches).toHaveBeenCalled(),
    );
    const body = reconcileGenerationBatches.mock.calls.at(-1)![1];
    expect(
      body.batch_ids?.length ?? body.client_batch_ids.length,
    ).toBeGreaterThan(0);
  });

  it("fences a replacement server instance instead of adopting its work", async () => {
    admitGenerationBatch.mockImplementation(
      (_target: unknown, body: { client_batch_id: string }) =>
        Promise.resolve(batch(body.client_batch_id)),
    );
    const stream = useGenerateStream();
    const id = stream.submit(request(), { kind: "single" }, route);
    await vi.waitFor(() =>
      expect(
        stream.jobs.value.find((job) => job.id === id)?.serverId,
      ).toBeTruthy(),
    );
    let confirmReplacement!: () => void;
    reconcileGenerationBatches.mockImplementation(
      () =>
        new Promise<GenerationBatchStatusResponse>((resolve) => {
          confirmReplacement = () =>
            resolve({
              ...statusResponse([]),
              instance_id: "replacement",
            });
        }),
    );

    __testing__.handleDurableEvent(
      route.hostId,
      "authority",
      JSON.stringify({ instance_id: "replacement" }),
    );

    expect(
      stream.jobs.value.find((candidate) => candidate.id === id)?.state,
    ).toBe("running");
    confirmReplacement();
    await vi.waitFor(() =>
      expect(
        stream.jobs.value.find((candidate) => candidate.id === id)?.state,
      ).toBe("error"),
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    expect(job.state).toBe("error");
    expect(job.detached).toBe(true);
    expect(job.error).toMatch(/replaced/i);
  });

  it("lets an observed completion beat a racing cancel and emits its effects once", async () => {
    admitGenerationBatch.mockImplementation(
      (_target: unknown, body: { client_batch_id: string }) =>
        Promise.resolve(batch(body.client_batch_id)),
    );
    let releaseMedia!: () => void;
    fetchGalleryBlob.mockImplementation(
      () =>
        new Promise<Blob>(
          (resolve) => (releaseMedia = () => resolve(new Blob(["media"]))),
        ),
    );
    const complete = vi.fn();
    const stream = useGenerateStream(complete);
    const id = stream.submit(request(), { kind: "single" }, route);
    await vi.waitFor(() =>
      expect(
        stream.jobs.value.find((job) => job.id === id)?.serverId,
      ).toBeTruthy(),
    );
    const clientBatchId = stream.jobs.value.find((job) => job.id === id)!
      .durableBatch!.clientBatchId;
    reconcileGenerationBatches.mockResolvedValue(
      statusResponse([batch(clientBatchId, ["complete"])]),
    );

    await __testing__.reconcileDurableHost(route.hostId);
    expect(stream.jobs.value.find((job) => job.id === id)?.state).toBe("done");
    await stream.cancel(id);
    expect(cancelQueueJob).not.toHaveBeenCalled();
    releaseMedia();
    await vi.waitFor(() => expect(complete).toHaveBeenCalledTimes(1));

    await __testing__.reconcileDurableHost(route.hostId);
    expect(complete).toHaveBeenCalledTimes(1);
    expect(fetchGalleryBlob).toHaveBeenCalledWith(
      expect.objectContaining({ id: route.hostId }),
      "print-1.png",
    );
  });

  it("rehydrates an unresolved durable record without dead-lettering or resubmitting", async () => {
    const persisted = {
      id: "local-1",
      request: request(),
      startedAt: 1,
      progress: {
        stage: "Queued",
        step: null,
        totalSteps: null,
        weightBytesLoaded: null,
        weightBytesTotal: null,
        queuePosition: null,
        gpu: null,
        elapsedMs: null,
      },
      result: null,
      error: null,
      state: "running" as const,
      chain: null,
      lastProgressAt: 1,
      workStarted: false,
      hostId: "origin",
      hostLabel: "this server",
      serverId: null,
      durableBatch: {
        clientBatchId: "client-reload",
        expectedInstanceId: "instance-1",
        serverBatchId: null,
        childIndex: 1,
      },
    };
    const loaded = __testing__.loadPersistedJobs(
      JSON.stringify({ version: 1, jobs: [persisted] }),
    );
    expect(loaded[0]!.state).toBe("running");
    expect(loaded[0]!.detached).toBe(true);

    const stream = useGenerateStream();
    stream.jobs.value = loaded as Job[];
    __testing__.resetDurableLifecycleForTests();
    reconcileGenerationBatches.mockResolvedValue(
      statusResponse([batch("client-reload")]),
    );
    useGenerateStream();

    await vi.waitFor(() =>
      expect(stream.jobs.value[0]!.serverId).toBe("job-client-reload-1"),
    );
    expect(admitGenerationBatch).not.toHaveBeenCalled();
  });

  it("finishes exact-filename media recovery after a reload during completion", async () => {
    const persisted = {
      id: "local-complete",
      request: request(),
      startedAt: 1,
      progress: {
        stage: "Developing",
        step: null,
        totalSteps: null,
        weightBytesLoaded: null,
        weightBytesTotal: null,
        queuePosition: null,
        gpu: null,
        elapsedMs: null,
      },
      result: null,
      error: null,
      state: "done" as const,
      chain: null,
      lastProgressAt: 20,
      workStarted: true,
      hostId: "origin",
      hostLabel: "this server",
      serverId: "job-client-complete-1",
      durableBatch: {
        clientBatchId: "client-complete",
        expectedInstanceId: "instance-1",
        serverBatchId: "server-client-complete",
        childIndex: 1,
      },
    };
    const stream = useGenerateStream();
    stream.jobs.value = __testing__.loadPersistedJobs(
      JSON.stringify({ version: 1, jobs: [persisted] }),
    ) as Job[];
    __testing__.resetDurableLifecycleForTests();
    reconcileGenerationBatches.mockResolvedValue(
      statusResponse([batch("client-complete", ["complete"])]),
    );

    useGenerateStream();

    await vi.waitFor(() =>
      expect(stream.jobs.value[0]!.result?.image).toBeTruthy(),
    );
    expect(fetchGalleryBlob).toHaveBeenCalledWith(
      expect.objectContaining({ id: "origin" }),
      "print-1.png",
    );
    expect(admitGenerationBatch).not.toHaveBeenCalled();
  });

  it.each([
    ["older host", { ...route, durableGeneration: null }, request()],
    ["identity", route, request("identity")],
    ["references", route, request("references")],
    [
      "MiniMax H3",
      route,
      { ...request(), model: "minimax-h3-fl2va:official-bf16" },
    ],
  ])(
    "keeps %s on the explicit legacy stream",
    async (_name, candidateRoute, candidate) => {
      if (_name === "identity") candidate.id_image = "private-face";
      if (_name === "references") candidate.references = [];
      const stream = useGenerateStream();
      stream.submit(candidate, { kind: "single" }, candidateRoute);
      await Promise.resolve();

      expect(generateStream).toHaveBeenCalledTimes(1);
      expect(admitGenerationBatch).not.toHaveBeenCalled();
    },
  );

  it.each(GENERATION_REQUEST_MEDIA_FIELDS)(
    "keeps media-bearing %s requests on the legacy stream",
    async (field) => {
      const valueByField: Record<string, unknown> = {
        source_image: "source",
        edit_images: ["edit"],
        references: [],
        id_image: "identity",
        id_images: ["identity"],
        mask_image: "mask",
        control_image: "control",
        audio_file: "audio",
        audio_file_path: "/audio",
        source_video: "source-video",
        source_video_path: "/source-video",
        extend_video: "extend-video",
        extend_video_path: "/extend-video",
        keyframes: [{ frame: 0, image: "keyframe" }],
        hdr_exr_dir: "/hdr",
      };
      const stream = useGenerateStream();

      stream.submit(
        { ...request(), [field]: valueByField[field] },
        { kind: "single" },
        route,
      );
      await Promise.resolve();

      expect(generateStream).toHaveBeenCalledTimes(1);
      expect(admitGenerationBatch).not.toHaveBeenCalled();
    },
  );

  it("keeps every waiting media job visible while draining four streams per target", async () => {
    const opened: Array<{
      prompt: string;
      target: string;
    }> = [];
    generateStream.mockImplementation(
      (
        candidate: GenerateRequestWire,
        _handlers: unknown,
        signal: AbortSignal,
        target: { baseUrl?: string } | undefined,
      ) =>
        new Promise<void>((resolve) => {
          opened.push({
            prompt: candidate.prompt,
            target: target?.baseUrl ?? "origin",
          });
          signal.addEventListener("abort", () => resolve(), { once: true });
        }),
    );
    const otherRoute: HostRoute = {
      ...route,
      hostId: "render-box-b",
      label: "Render box B",
      target: { baseUrl: "http://render-box-b:7680", apiKey: "secret-b" },
      instanceId: "instance-2",
    };
    const stream = useGenerateStream();
    const ids = [route, otherRoute].flatMap((candidateRoute) =>
      Array.from({ length: 5 }, (_, index) =>
        stream.submit(
          {
            ...request(`${candidateRoute.hostId}-${index}`),
            source_image: `session-media-${candidateRoute.hostId}-${index}`,
          },
          { kind: "single" },
          candidateRoute,
        ),
      ),
    );

    try {
      expect(
        stream.jobs.value.filter((job) => ids.includes(job.id)),
      ).toHaveLength(10);
      expect(
        opened.filter((entry) => entry.target === route.target.baseUrl),
      ).toHaveLength(4);
      expect(
        opened.filter((entry) => entry.target === otherRoute.target.baseUrl),
      ).toHaveLength(4);

      const waitingA = stream.jobs.value.find(
        (job) => job.request.prompt === "render-box-4",
      )!;
      const waitingB = stream.jobs.value.find(
        (job) => job.request.prompt === "render-box-b-4",
      )!;
      expect(waitingA.streamStarted).toBe(false);
      expect(waitingB.streamStarted).toBe(false);

      await stream.cancel(waitingA.id);
      expect(waitingA.state).toBe("canceled");
      stream.jobs.value
        .find((job) => job.request.prompt === "render-box-0")!
        .controller.abort();
      await Promise.resolve();
      expect(
        opened.filter((entry) => entry.target === route.target.baseUrl),
      ).toHaveLength(4);

      const activeB = stream.jobs.value.find(
        (job) => job.request.prompt === "render-box-b-0",
      )!;
      activeB.controller.abort();
      await vi.waitFor(() =>
        expect(
          opened.filter((entry) => entry.target === otherRoute.target.baseUrl),
        ).toHaveLength(5),
      );
      expect(waitingB.streamStarted).toBe(true);
    } finally {
      for (const job of stream.jobs.value.filter((candidate) =>
        ids.includes(candidate.id),
      )) {
        job.controller.abort();
        stream.remove(job.id);
      }
    }
  });
});
