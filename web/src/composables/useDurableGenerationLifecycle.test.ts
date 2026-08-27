import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GenerateRequestWire, GalleryImage } from "../types";
import type { HostRoute } from "../lib/hostRouting";
import { GENERATION_REQUEST_MEDIA_FIELDS } from "../lib/generationRequestMedia";
import type {
  GenerationBatchStatus,
  GenerationBatchStatusResponse,
} from "@studio/api/generationAdmission";
import { ApiError } from "@studio/api/client";

const admitGenerationBatch = vi.hoisted(() => vi.fn());
const lookupGenerationBatchByClientId = vi.hoisted(() => vi.fn());
const reconcileGenerationBatches = vi.hoisted(() => vi.fn());
const fetchEventSource = vi.hoisted(() => vi.fn(() => new Promise(() => {})));
const generateStream = vi.hoisted(() => vi.fn().mockResolvedValue(undefined));
const cancelQueueJob = vi.hoisted(() => vi.fn().mockResolvedValue(undefined));
const mutateQueueJobOnExpectedInstance = vi.hoisted(() =>
  vi.fn().mockResolvedValue(undefined),
);
const retryQueueJobRecoveringAmbiguity = vi.hoisted(() =>
  vi.fn().mockResolvedValue({ kind: "accepted" }),
);
const listGalleryFrom = vi.hoisted(() => vi.fn());
const fetchGalleryBlob = vi.hoisted(() => vi.fn());
const fetchGalleryThumbnailBlob = vi.hoisted(() => vi.fn());

vi.mock("@studio/api/generationAdmission", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/generationAdmission")>()),
  admitGenerationBatch,
  lookupGenerationBatchByClientId,
  reconcileGenerationBatches,
}));

vi.mock("@microsoft/fetch-event-source", () => ({ fetchEventSource }));

vi.mock("@studio/api/queuePlan", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/queuePlan")>()),
  mutateQueueJobOnExpectedInstance,
  retryQueueJobRecoveringAmbiguity,
}));

vi.mock("../api", () => ({
  cancelQueueJob,
  fetchQueue: vi.fn().mockResolvedValue({ entries: [] }),
  generateStream,
  generateChainStream: vi.fn().mockResolvedValue(undefined),
  listGalleryFrom,
}));

vi.mock("../lib/galleryMedia", () => ({
  fetchGalleryBlob,
  fetchGalleryThumbnailBlob,
}));

import { __testing__, useGenerateStream, type Job } from "./useGenerateStream";

const nativeStorageSetItem = localStorage.setItem;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

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

// Every shipping machine advertises both halves of the durable contract; the
// client gates on their presence and nothing about the request.
const route: HostRoute = {
  hostId: "render-box",
  label: "Render box",
  target: { baseUrl: "http://render-box:7680", apiKey: "secret" },
  instanceId: "instance-1",
  durableGeneration: {
    heterogeneous_batch_max_outputs: 64,
  },
  durableMedia: {
    protocol_version: 2,
    encrypted_at_rest: true,
    generate_request_media: true,
    identity: true,
    h3_references: false,
    private_h3: false,
  },
  eventsAvailable: true,
};

const durableMediaRoute: HostRoute = {
  ...route,
  durableMedia: {
    protocol_version: 2,
    encrypted_at_rest: true,
    generate_request_media: true,
    identity: true,
    h3_references: false,
    private_h3: false,
  },
};

const canonicalH3Route: HostRoute = {
  ...route,
  modelFamily: "minimax-h3",
  durableGeneration: {
    ...route.durableGeneration,
  },
  durableMedia: {
    protocol_version: 3,
    encrypted_at_rest: true,
    generate_request_media: true,
    identity: true,
    h3_references: true,
    private_h3: true,
  },
};

function batch(
  clientBatchId: string,
  states: Array<
    | "accepted"
    | "queued"
    | "running"
    | "held"
    | "complete"
    | "failed"
    | "cancelled"
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

function wavBlob(sampleRate = 24_000, channels = 2, frames = 24): Blob {
  const bytesPerSample = 2;
  const dataBytes = frames * channels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  const text = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, bytesPerSample * 8, true);
  text(36, "data");
  view.setUint32(40, dataBytes, true);
  return new Blob([buffer], { type: "audio/wav" });
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
  mutateQueueJobOnExpectedInstance.mockClear();
  retryQueueJobRecoveringAmbiguity.mockReset();
  retryQueueJobRecoveringAmbiguity.mockResolvedValue({ kind: "accepted" });
  listGalleryFrom.mockReset();
  fetchGalleryBlob.mockReset();
  fetchGalleryThumbnailBlob.mockReset();
  listGalleryFrom.mockResolvedValue([
    gallery("print-1.png"),
    gallery("print-2.png"),
  ]);
  fetchGalleryBlob.mockResolvedValue(new Blob(["media"]));
  fetchGalleryThumbnailBlob.mockResolvedValue(new Blob(["thumbnail"]));
  lookupGenerationBatchByClientId.mockResolvedValue({ kind: "missing" });
});

describe("web durable generation lifecycle", () => {
  it("admits supported media without opening a held generation stream or persisting its bytes", () => {
    admitGenerationBatch.mockImplementation(() => new Promise(() => {}));
    const stream = useGenerateStream();

    const submitted = request("encrypted media print");
    submitted.source_image = "PRIVATE-DURABLE-SOURCE";
    stream.submit(submitted, { kind: "single" }, durableMediaRoute);

    expect(admitGenerationBatch).toHaveBeenCalledTimes(1);
    expect(admitGenerationBatch.mock.calls[0]![1].requests[0]).toMatchObject({
      source_image: "PRIVATE-DURABLE-SOURCE",
    });
    expect(generateStream).not.toHaveBeenCalled();
    const persisted = Array.from({ length: localStorage.length }, (_, index) =>
      localStorage.getItem(localStorage.key(index)!),
    ).join("\n");
    expect(persisted).not.toContain("PRIVATE-DURABLE-SOURCE");
  });

  it("admits an opaque H3 family through the same durable batch", () => {
    admitGenerationBatch.mockImplementation(() => new Promise(() => {}));
    const stream = useGenerateStream();
    const submitted = request("opaque H3");
    submitted.model = "hf:opaque-h3-checkpoint";
    submitted.source_image = "PRIVATE-H3-SOURCE";

    stream.submit(
      submitted,
      { kind: "single" },
      {
        ...durableMediaRoute,
        modelFamily: "minimax-h3",
      },
    );

    expect(admitGenerationBatch).toHaveBeenCalledTimes(1);
    expect(generateStream).not.toHaveBeenCalled();
  });

  it("admits canonical v2 H3 through the durable batch transport", () => {
    admitGenerationBatch.mockImplementation(() => new Promise(() => {}));
    const stream = useGenerateStream();
    const submitted = request("canonical H3");
    submitted.model = "hf:opaque-h3-checkpoint";
    submitted.source_image = "PRIVATE-H3-SOURCE";

    stream.submit(submitted, { kind: "single" }, canonicalH3Route);

    expect(admitGenerationBatch).toHaveBeenCalledTimes(1);
    expect(admitGenerationBatch.mock.calls[0]![1].requests[0]).toMatchObject({
      model: "hf:opaque-h3-checkpoint",
      source_image: "PRIVATE-H3-SOURCE",
    });
    expect(generateStream).not.toHaveBeenCalled();
  });

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

  it("chunks Batch N at the host limit without opening attached streams", () => {
    admitGenerationBatch.mockImplementation(() => new Promise(() => {}));
    const limitedRoute: HostRoute = {
      ...route,
      durableGeneration: {
        ...route.durableGeneration,
        heterogeneous_batch_max_outputs: 2,
      },
    };
    const stream = useGenerateStream();

    const ids = stream.submitBatch(
      [request("one"), request("two"), request("three")],
      { kind: "single" },
      limitedRoute,
    );

    expect(ids).toHaveLength(3);
    expect(admitGenerationBatch).toHaveBeenCalledTimes(2);
    expect(
      admitGenerationBatch.mock.calls.map((call) => call[1].requests.length),
    ).toEqual([2, 1]);
    expect(generateStream).not.toHaveBeenCalled();
  });

  it("never turns browser quota into a durable admission gate", async () => {
    admitGenerationBatch.mockImplementation(() => new Promise(() => {}));
    const storage = vi
      .spyOn(localStorage, "setItem")
      .mockImplementation(function (key, value) {
        if (key.startsWith("mold.generate.jobs")) {
          throw new DOMException("quota exceeded", "QuotaExceededError");
        }
        return Reflect.apply(nativeStorageSetItem, localStorage, [key, value]);
      });
    try {
      const stream = useGenerateStream();

      const id = stream.submit(request(), { kind: "single" }, route);
      await Promise.resolve();

      const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
      expect(job.state).toBe("running");
      expect(job.error).toBeNull();
      expect(job.durableBatch).toBeDefined();
      expect(admitGenerationBatch).toHaveBeenCalledTimes(1);
      expect(generateStream).not.toHaveBeenCalled();
      expect(lookupGenerationBatchByClientId).not.toHaveBeenCalled();
      expect(fetchEventSource).toHaveBeenCalledTimes(1);
    } finally {
      storage.mockRestore();
    }
  });

  it.each([
    ["commit-then-500", new ApiError("response lost", 500)],
    ["disconnect", new TypeError("connection closed")],
  ])(
    "recovers an ambiguous %s POST only by UUID and never falls back",
    async (_case, failure) => {
      let generationWrites = 0;
      const storage = vi
        .spyOn(localStorage, "setItem")
        .mockImplementation(function (key, value) {
          if (key === "mold.generate.jobs" && generationWrites++ > 0) {
            throw new DOMException("quota exceeded", "QuotaExceededError");
          }
          return Reflect.apply(nativeStorageSetItem, localStorage, [
            key,
            value,
          ]);
        });
      try {
        admitGenerationBatch.mockRejectedValue(failure);
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
        const clientBatchId = stream.jobs.value.find((job) => job.id === id)!
          .durableBatch!.clientBatchId;
        const persisted = localStorage.getItem(
          `mold.generate.jobs.recovery.${clientBatchId}`,
        )!;
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
    },
  );

  it("persists each deep outstanding batch independently instead of rewriting the backlog", () => {
    admitGenerationBatch.mockImplementation(() => new Promise(() => {}));
    const writes: Array<{ key: string; bytes: number }> = [];
    const storage = vi
      .spyOn(localStorage, "setItem")
      .mockImplementation(function (key, value) {
        writes.push({ key, bytes: value.length });
        return Reflect.apply(nativeStorageSetItem, localStorage, [key, value]);
      });
    try {
      const stream = useGenerateStream();
      for (let index = 0; index < 250; index += 1) {
        stream.submit(
          request(`deep print ${index}`),
          { kind: "single" },
          route,
        );
      }

      expect(admitGenerationBatch).toHaveBeenCalledTimes(250);
      expect(
        stream.jobs.value.filter((job) => job.state === "running"),
      ).toHaveLength(250);
      expect(writes.filter(({ key }) => key === "mold.generate.jobs")).toEqual(
        [],
      );
      expect(localStorage.getItem("mold.generate.jobs")).toBeNull();
      const recoveryWrites = writes.filter(({ key }) =>
        key.startsWith("mold.generate.jobs.recovery."),
      );
      expect(recoveryWrites).toHaveLength(250);
      expect(new Set(recoveryWrites.map(({ key }) => key)).size).toBe(250);
      expect(
        recoveryWrites.every(({ key }) => {
          const record = JSON.parse(localStorage.getItem(key)!);
          return record.jobs.length === 1;
        }),
      ).toBe(true);
      const recovered = __testing__.loadDurableRecoveryJobs(localStorage);
      expect(recovered).toHaveLength(250);
      expect(
        recovered.every(
          (job) => job.state === "running" && job.detached === true,
        ),
      ).toBe(true);
    } finally {
      storage.mockRestore();
    }
  });

  it("records cancel intent before admission resolves and deletes the exact admitted job", async () => {
    let confirmAdmission!: (value: GenerationBatchStatus) => void;
    admitGenerationBatch.mockImplementation(
      (_target: unknown, body: { client_batch_id: string }) =>
        new Promise<GenerationBatchStatus>((resolve) => {
          confirmAdmission = (value) => resolve(value);
          void body;
        }),
    );
    const stream = useGenerateStream();
    const id = stream.submit(
      request("cancel immediately"),
      { kind: "single" },
      route,
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    const clientBatchId = job.durableBatch!.clientBatchId;

    await stream.cancel(id);
    expect(job.state).toBe("running");
    expect(job.cancelRequested).toBe(true);
    expect(mutateQueueJobOnExpectedInstance).not.toHaveBeenCalled();
    expect(
      localStorage.getItem(`mold.generate.jobs.recovery.${clientBatchId}`),
    ).toContain('"cancelRequested":true');

    confirmAdmission(batch(clientBatchId));
    await vi.waitFor(() =>
      expect(mutateQueueJobOnExpectedInstance).toHaveBeenCalledWith(
        route.target,
        {
          instanceId: route.instanceId,
          batchId: `server-${clientBatchId}`,
          clientBatchId,
          jobId: `job-${clientBatchId}-1`,
        },
        "cancel",
      ),
    );
    await vi.waitFor(() => expect(job.state).toBe("canceled"));
  });

  it("carries early cancel intent through an ambiguous admission lookup", async () => {
    let recover!: (value: {
      kind: "found";
      batch: GenerationBatchStatus;
    }) => void;
    admitGenerationBatch.mockRejectedValue(new TypeError("response lost"));
    lookupGenerationBatchByClientId.mockImplementation(
      () =>
        new Promise((resolve) => {
          recover = resolve;
        }),
    );
    const stream = useGenerateStream();
    const id = stream.submit(
      request("ambiguous cancel"),
      { kind: "single" },
      route,
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    const clientBatchId = job.durableBatch!.clientBatchId;

    await stream.cancel(id);
    await vi.waitFor(() =>
      expect(lookupGenerationBatchByClientId).toHaveBeenCalled(),
    );
    recover({ kind: "found", batch: batch(clientBatchId) });

    await vi.waitFor(() =>
      expect(mutateQueueJobOnExpectedInstance).toHaveBeenCalledWith(
        route.target,
        {
          instanceId: route.instanceId,
          batchId: `server-${clientBatchId}`,
          clientBatchId,
          jobId: `job-${clientBatchId}-1`,
        },
        "cancel",
      ),
    );
    await vi.waitFor(() => expect(job.state).toBe("canceled"));
  });

  it("announces a held child once, with the machine's own reason", async () => {
    // A print is admitted BEFORE its model is resolved, so "nobody has this
    // model" arrives as a held child rather than an infeasible preview. The
    // pull offer hangs off this listener; without it the pull is lost.
    admitGenerationBatch.mockImplementation(
      (_target: unknown, body: { client_batch_id: string }) =>
        Promise.resolve(batch(body.client_batch_id)),
    );
    const held: GenerationBatchStatus[] = [];
    const stream = useGenerateStream(undefined, (job) =>
      held.push(job as never),
    );
    const id = stream.submit(request("held print"), { kind: "single" }, route);
    await vi.waitFor(() =>
      expect(
        stream.jobs.value.find((job) => job.id === id)?.serverId,
      ).toBeTruthy(),
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    const clientBatchId = job.durableBatch!.clientBatchId;
    const parked = batch(clientBatchId, ["held"]);
    parked.children[0] = {
      ...parked.children[0]!,
      error:
        "deferred generation preparation failed: model 'flux-dev:q8' is not downloaded",
      error_code: "UNKNOWN_MODEL",
      retryable: true,
    };
    reconcileGenerationBatches.mockResolvedValue(statusResponse([parked]));

    await __testing__.reconcileDurableHost(route.hostId);
    // A second reconciliation of the same hold must not re-announce it.
    await __testing__.reconcileDurableHost(route.hostId);

    expect(held).toHaveLength(1);
    expect(job.holdError).toBe(
      "deferred generation preparation failed: model 'flux-dev:q8' is not downloaded",
    );
    expect(job.holdCode).toBe("UNKNOWN_MODEL");
    expect(job.retryable).toBe(true);
  });

  it("retries held durable work only through the admitting instance fence", async () => {
    admitGenerationBatch.mockImplementation(
      (_target: unknown, body: { client_batch_id: string }) =>
        Promise.resolve(batch(body.client_batch_id)),
    );
    const stream = useGenerateStream();
    const id = stream.submit(request("retry held"), { kind: "single" }, route);
    await vi.waitFor(() =>
      expect(
        stream.jobs.value.find((job) => job.id === id)?.serverId,
      ).toBeTruthy(),
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    const clientBatchId = job.durableBatch!.clientBatchId;
    const held = batch(clientBatchId, ["held"]);
    held.children[0] = {
      ...held.children[0]!,
      error: "model dependency is unavailable",
      retryable: true,
    };
    reconcileGenerationBatches.mockResolvedValue(statusResponse([held]));
    await __testing__.reconcileDurableHost(route.hostId);

    const confirmation = deferred<{ kind: "accepted" }>();
    retryQueueJobRecoveringAmbiguity.mockReturnValue(confirmation.promise);
    const retry = stream.retry(id);

    expect(job).toMatchObject({ retryable: false, retrying: true });
    confirmation.resolve({ kind: "accepted" });
    await retry;

    expect(retryQueueJobRecoveringAmbiguity).toHaveBeenCalledWith(
      route.target,
      {
        instanceId: route.instanceId,
        batchId: `server-${clientBatchId}`,
        clientBatchId,
        jobId: `job-${clientBatchId}-1`,
      },
    );
  });

  it("keeps recovery-record media redaction as defense in depth", () => {
    const redacted = __testing__.durablePersistenceSafeRequest({
      ...request(),
      source_image: "source",
      source_image_name: "private-source-name.png",
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
    expect(redacted).not.toHaveProperty("source_image_name");
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

  it("reconciles again from the post-commit hint when earlier terminal hints race", async () => {
    admitGenerationBatch.mockImplementation(
      (_target: unknown, body: { client_batch_id: string }) =>
        Promise.resolve(batch(body.client_batch_id)),
    );
    const completed = vi.fn();
    const stream = useGenerateStream(completed);
    const id = stream.submit(request("post-commit"), { kind: "single" }, route);
    await vi.waitFor(() =>
      expect(
        stream.jobs.value.find((job) => job.id === id)?.serverId,
      ).toBeTruthy(),
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    const clientBatchId = job.durableBatch!.clientBatchId;
    reconcileGenerationBatches
      .mockResolvedValueOnce(
        statusResponse([batch(clientBatchId, ["running"])]),
      )
      .mockResolvedValueOnce(
        statusResponse([batch(clientBatchId, ["complete"])]),
      );

    __testing__.handleDurableEvent(
      route.hostId,
      "event",
      JSON.stringify({ type: "job_ended", id: job.serverId }),
    );
    await vi.waitFor(() =>
      expect(reconcileGenerationBatches).toHaveBeenCalledTimes(1),
    );
    expect(job.state).toBe("running");

    __testing__.handleDurableEvent(
      route.hostId,
      "event",
      JSON.stringify({
        type: "job_state_committed",
        id: "committed-before-client-map",
      }),
    );

    await vi.waitFor(() => expect(job.state).toBe("done"));
    await vi.waitFor(() => expect(completed).toHaveBeenCalledTimes(1));
  });

  it("scopes child commits and reserves host-wide reads for bulk commits", async () => {
    admitGenerationBatch.mockImplementation(
      (_target: unknown, body: { client_batch_id: string }) =>
        Promise.resolve(batch(body.client_batch_id)),
    );
    const stream = useGenerateStream();
    const firstId = stream.submit(request("first"), { kind: "single" }, route);
    const secondId = stream.submit(
      request("second"),
      { kind: "single" },
      route,
    );
    await vi.waitFor(() =>
      expect(
        stream.jobs.value.filter(
          (job) => (job.id === firstId || job.id === secondId) && job.serverId,
        ),
      ).toHaveLength(2),
    );
    const first = stream.jobs.value.find((job) => job.id === firstId)!;
    const second = stream.jobs.value.find((job) => job.id === secondId)!;
    reconcileGenerationBatches.mockImplementation(
      (_target: unknown, body: { batch_ids?: string[] }) => {
        const requested = new Set(body.batch_ids ?? []);
        return Promise.resolve(
          statusResponse(
            [first, second]
              .filter((job) => requested.has(job.durableBatch!.serverBatchId!))
              .map((job) =>
                batch(job.durableBatch!.clientBatchId, ["running"]),
              ),
          ),
        );
      },
    );
    reconcileGenerationBatches.mockClear();

    __testing__.handleDurableEvent(
      route.hostId,
      "event",
      JSON.stringify({ type: "job_state_committed", id: first.serverId }),
    );
    await vi.waitFor(() =>
      expect(reconcileGenerationBatches).toHaveBeenCalledTimes(1),
    );
    expect(reconcileGenerationBatches.mock.calls[0]![1].batch_ids).toEqual([
      first.durableBatch!.serverBatchId,
    ]);

    reconcileGenerationBatches.mockClear();
    __testing__.handleDurableEvent(
      route.hostId,
      "event",
      JSON.stringify({ type: "generation_states_committed" }),
    );
    await vi.waitFor(() =>
      expect(reconcileGenerationBatches).toHaveBeenCalledTimes(1),
    );
    expect(
      new Set(reconcileGenerationBatches.mock.calls[0]![1].batch_ids),
    ).toEqual(
      new Set([
        first.durableBatch!.serverBatchId,
        second.durableBatch!.serverBatchId,
      ]),
    );
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
    expect(mutateQueueJobOnExpectedInstance).not.toHaveBeenCalled();
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
    {
      name: "MP4",
      filename: "print-1.mp4",
      format: "mp4" as const,
      artifact: new Blob(["video"], { type: "video/mp4" }),
    },
    {
      name: "WAV",
      filename: "print-1.wav",
      format: "wav" as const,
      artifact: wavBlob(),
    },
  ])(
    "hydrates durable $name completion as typed playable media",
    async (media) => {
      admitGenerationBatch.mockImplementation(
        (_target: unknown, body: { client_batch_id: string }) =>
          Promise.resolve(batch(body.client_batch_id)),
      );
      const row = {
        ...gallery(media.filename),
        format: media.format,
        metadata: {
          ...gallery(media.filename).metadata,
          output_format: media.format,
          frames: media.format === "mp4" ? 25 : null,
          fps: media.format === "mp4" ? 24 : null,
        },
      } satisfies GalleryImage;
      listGalleryFrom.mockResolvedValue([row]);
      fetchGalleryBlob.mockResolvedValue(media.artifact);
      const stream = useGenerateStream();
      const id = stream.submit(request(media.name), { kind: "single" }, route);
      await vi.waitFor(() =>
        expect(
          stream.jobs.value.find((job) => job.id === id)?.serverId,
        ).toBeTruthy(),
      );
      const clientBatchId = stream.jobs.value.find((job) => job.id === id)!
        .durableBatch!.clientBatchId;
      reconcileGenerationBatches.mockResolvedValue(
        statusResponse([
          batch(clientBatchId, ["complete"], {
            children: [
              {
                ...batch(clientBatchId, ["complete"]).children[0]!,
                result: { filename: media.filename },
              },
            ],
          }),
        ]),
      );

      await __testing__.reconcileDurableHost(route.hostId);
      await vi.waitFor(() =>
        expect(
          stream.jobs.value.find((job) => job.id === id)?.result,
        ).toBeTruthy(),
      );
      const result = stream.jobs.value.find((job) => job.id === id)!.result!;
      expect(result.format).toBe(media.format);
      if (media.format === "mp4") {
        expect(result.video_frames).toBe(25);
        expect(result.video_fps).toBe(24);
        expect(result.video_thumbnail).toBeTruthy();
        expect(result.audio_sample_rate).toBeUndefined();
      } else {
        expect(result.audio_sample_rate).toBe(24_000);
        expect(result.audio_channels).toBe(2);
        expect(result.audio_duration_ms).toBe(1);
        expect(result.audio_thumbnail).toBeTruthy();
        expect(result.video_frames).toBeUndefined();
      }
    },
  );

  it("shares one gallery snapshot and bounds exact artifact reads across a completion wave", async () => {
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
    listGalleryFrom.mockResolvedValue(
      Array.from({ length: 8 }, (_, index) =>
        gallery(`print-${index + 1}.png`),
      ),
    );
    let releaseReads!: () => void;
    const gate = new Promise<void>((resolve) => (releaseReads = resolve));
    let activeReads = 0;
    let maxActiveReads = 0;
    fetchGalleryBlob.mockImplementation(async () => {
      activeReads += 1;
      maxActiveReads = Math.max(maxActiveReads, activeReads);
      await gate;
      activeReads -= 1;
      return new Blob(["media"]);
    });
    const completed = vi.fn();
    const stream = useGenerateStream(completed);
    const ids = stream.submitBatch(
      Array.from({ length: 8 }, (_, index) => request(`wave ${index}`)),
      { kind: "single" },
      route,
    );
    await vi.waitFor(() =>
      expect(
        ids.every(
          (id) => stream.jobs.value.find((job) => job.id === id)?.serverId,
        ),
      ).toBe(true),
    );
    const clientBatchId = stream.jobs.value.find((job) => job.id === ids[0])!
      .durableBatch!.clientBatchId;
    reconcileGenerationBatches.mockResolvedValue(
      statusResponse([
        batch(
          clientBatchId,
          Array.from({ length: 8 }, () => "complete"),
        ),
      ]),
    );

    await __testing__.reconcileDurableHost(route.hostId);
    await vi.waitFor(() => expect(fetchGalleryBlob).toHaveBeenCalledTimes(4));
    expect(maxActiveReads).toBe(4);
    expect(listGalleryFrom).toHaveBeenCalledTimes(1);
    releaseReads();
    await vi.waitFor(() => expect(completed).toHaveBeenCalledTimes(8));
    expect(maxActiveReads).toBe(4);
    expect(listGalleryFrom).toHaveBeenCalledTimes(1);
  });

  it("uses an exact gallery event row without launching a gallery listing", async () => {
    admitGenerationBatch.mockImplementation(
      (_target: unknown, body: { client_batch_id: string }) =>
        Promise.resolve(batch(body.client_batch_id)),
    );
    const stream = useGenerateStream();
    const id = stream.submit(request("exact event"), { kind: "single" }, route);
    await vi.waitFor(() =>
      expect(
        stream.jobs.value.find((job) => job.id === id)?.serverId,
      ).toBeTruthy(),
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    const clientBatchId = job.durableBatch!.clientBatchId;
    const image = gallery("print-1.png");
    image.metadata.job_id = job.serverId;
    reconcileGenerationBatches.mockResolvedValue(
      statusResponse([batch(clientBatchId, ["complete"])]),
    );

    __testing__.handleDurableEvent(
      route.hostId,
      "event",
      JSON.stringify({
        type: "gallery_added",
        filename: image.filename,
        image,
      }),
    );

    await vi.waitFor(() =>
      expect(
        stream.jobs.value.find((candidate) => candidate.id === id)?.result,
      ).toBeTruthy(),
    );
    expect(listGalleryFrom).not.toHaveBeenCalled();
    expect(fetchGalleryBlob).toHaveBeenCalledWith(
      expect.objectContaining({ id: route.hostId }),
      "print-1.png",
    );
    expect(reconcileGenerationBatches.mock.calls[0]![1]).toMatchObject({
      client_batch_ids: [],
      batch_ids: [job.durableBatch!.serverBatchId],
    });
  });

  it("keeps complete authority and retries after a transient artifact read failure", async () => {
    admitGenerationBatch.mockImplementation(
      (_target: unknown, body: { client_batch_id: string }) =>
        Promise.resolve(batch(body.client_batch_id)),
    );
    fetchGalleryBlob
      .mockRejectedValueOnce(new TypeError("temporary read failure"))
      .mockResolvedValueOnce(new Blob(["media"]));
    const stream = useGenerateStream();
    const id = stream.submit(request("retry media"), { kind: "single" }, route);
    await vi.waitFor(() =>
      expect(
        stream.jobs.value.find((job) => job.id === id)?.serverId,
      ).toBeTruthy(),
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    const clientBatchId = job.durableBatch!.clientBatchId;
    reconcileGenerationBatches.mockResolvedValue(
      statusResponse([batch(clientBatchId, ["complete"])]),
    );

    await __testing__.reconcileDurableHost(route.hostId);
    await vi.waitFor(() =>
      expect(job.mediaHydrationError).toMatch(/temporary/),
    );
    expect(job.state).toBe("done");
    expect(job.result).toBeNull();

    await __testing__.reconcileDurableHost(route.hostId);
    await vi.waitFor(() => expect(job.result?.image).toBeTruthy());
    expect(job.state).toBe("done");
    expect(job.mediaHydrationError).toBeNull();
    expect(fetchGalleryBlob).toHaveBeenCalledTimes(2);
  });

  it("targets an exact batch for ordinary lifecycle hints", async () => {
    admitGenerationBatch.mockImplementation(
      (_target: unknown, body: { client_batch_id: string }) =>
        Promise.resolve(batch(body.client_batch_id)),
    );
    const stream = useGenerateStream();
    const firstId = stream.submit(request("first"), { kind: "single" }, route);
    const secondId = stream.submit(
      request("second"),
      { kind: "single" },
      route,
    );
    await vi.waitFor(() =>
      expect(
        [firstId, secondId].every(
          (id) => stream.jobs.value.find((job) => job.id === id)?.serverId,
        ),
      ).toBe(true),
    );
    const first = stream.jobs.value.find((job) => job.id === firstId)!;
    const second = stream.jobs.value.find((job) => job.id === secondId)!;
    reconcileGenerationBatches
      .mockReset()
      .mockResolvedValue(
        statusResponse([batch(first.durableBatch!.clientBatchId)]),
      );

    __testing__.handleDurableEvent(
      route.hostId,
      "event",
      JSON.stringify({ type: "job_started", id: first.serverId }),
    );

    await vi.waitFor(() =>
      expect(reconcileGenerationBatches).toHaveBeenCalledTimes(1),
    );
    const requestBody = reconcileGenerationBatches.mock.calls[0]![1];
    expect(requestBody.batch_ids).toEqual([first.durableBatch!.serverBatchId]);
    expect(requestBody.batch_ids).not.toContain(
      second.durableBatch!.serverBatchId,
    );
  });

  it.each([
    [
      "a machine with no durable queue",
      { ...route, durableGeneration: null },
      "does not advertise the durable generation queue",
    ],
    [
      "a machine with no durable request media",
      { ...route, durableMedia: null },
      "does not advertise durable request media",
    ],
  ])(
    "refuses %s by name and queues nothing",
    (_name, candidateRoute, reason) => {
      const stream = useGenerateStream();

      expect(() =>
        stream.submit(request(), { kind: "single" }, candidateRoute),
      ).toThrow(reason);
      expect(admitGenerationBatch).not.toHaveBeenCalled();
      expect(generateStream).not.toHaveBeenCalled();
      expect(stream.jobs.value).toHaveLength(0);
    },
  );

  /**
   * The durable protocol carries every request trait. A client-side per-trait
   * fence could only refuse work the server would have taken, so the server's
   * own typed admission refusal is the single authority.
   */
  it.each(GENERATION_REQUEST_MEDIA_FIELDS)(
    "admits a media-bearing %s request unchanged",
    (field) => {
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
      admitGenerationBatch.mockImplementation(() => new Promise(() => {}));
      const stream = useGenerateStream();

      stream.submit(
        { ...request(), [field]: valueByField[field] },
        { kind: "single" },
        durableMediaRoute,
      );

      expect(admitGenerationBatch).toHaveBeenCalledTimes(1);
      expect(admitGenerationBatch.mock.calls[0]![1].requests[0]).toMatchObject({
        [field]: valueByField[field],
      });
      expect(generateStream).not.toHaveBeenCalled();
    },
  );

  it("admits media combined with a LoRA, and an HDR directory, unchanged", () => {
    admitGenerationBatch.mockImplementation(() => new Promise(() => {}));
    const stream = useGenerateStream();

    stream.submit(
      {
        ...request("lora + media"),
        source_image: "source",
        loras: [{ path: "local", scale: 1 }],
      },
      { kind: "single" },
      durableMediaRoute,
    );
    stream.submit(
      {
        ...request("hdr"),
        hdr_exr_dir: "/private/hdr",
      } as unknown as GenerateRequestWire,
      { kind: "single" },
      durableMediaRoute,
    );

    expect(admitGenerationBatch).toHaveBeenCalledTimes(2);
    expect(admitGenerationBatch.mock.calls[0]![1].requests[0]).toMatchObject({
      source_image: "source",
      loras: [{ path: "local", scale: 1 }],
    });
    expect(admitGenerationBatch.mock.calls[1]![1].requests[0]).toMatchObject({
      hdr_exr_dir: "/private/hdr",
    });
    expect(generateStream).not.toHaveBeenCalled();
  });

  it("admits every media print immediately — there is no browser stream budget left to drain", () => {
    admitGenerationBatch.mockImplementation(() => new Promise(() => {}));
    const otherRoute: HostRoute = {
      ...durableMediaRoute,
      hostId: "render-box-b",
      label: "Render box B",
      target: { baseUrl: "http://render-box-b:7680", apiKey: "secret-b" },
      instanceId: "instance-2",
    };
    const stream = useGenerateStream();
    const ids = [durableMediaRoute, otherRoute].flatMap((candidateRoute) =>
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

    expect(
      stream.jobs.value.filter((job) => ids.includes(job.id)),
    ).toHaveLength(10);
    expect(admitGenerationBatch).toHaveBeenCalledTimes(10);
    expect(generateStream).not.toHaveBeenCalled();
  });
});
