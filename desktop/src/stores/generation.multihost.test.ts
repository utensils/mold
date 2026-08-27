import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises } from "@vue/test-utils";
import type { GenerateRequest } from "../lib/api/types";

const sseStream = vi.fn();
vi.mock("../lib/api/sse", () => ({
  sseStream: (...a: unknown[]) => sseStream(...a),
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

const streamableMediaUrl = vi.fn().mockResolvedValue("https://hal9000/media/generated-video");
const evictMedia = vi.fn();
vi.mock("../lib/gallery/media", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/gallery/media")>()),
  streamableMediaUrl: (...a: unknown[]) => streamableMediaUrl(...a),
  evictMedia: (...a: unknown[]) => evictMedia(...a),
}));

/** The live primary connection, re-pointable mid-test. */
const primary = vi.hoisted(() => ({
  target: { baseUrl: "http://primary:7680", apiKey: "pk" } as {
    baseUrl: string;
    apiKey: string | null;
  } | null,
}));
const apiFetchTo = vi.fn().mockResolvedValue(new Response(null, { status: 200 }));
const apiJsonTo = vi.fn().mockResolvedValue([]);
vi.mock("../lib/api/client", () => ({
  ApiError: class ApiError extends Error {
    constructor(
      message: string,
      public readonly status: number,
    ) {
      super(message);
    }
  },
  apiFetchTo: (...a: unknown[]) => apiFetchTo(...a),
  apiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
  conditionalApiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
  currentTarget: () => {
    if (!primary.target) throw new Error("No engine connected.");
    return primary.target;
  },
}));

vi.mock("../lib/notify", () => ({
  notifyGenerated: vi.fn(),
  notifyGenerationFailed: vi.fn(),
}));

const saveOutputBytes = vi.fn().mockResolvedValue("saved.png");
const localGalleryList = vi.fn().mockResolvedValue({ images: [], target: null });
vi.mock("../lib/ipc", () => ({
  inTauri: () => true,
  ipc: {
    saveOutputBytes: (...a: unknown[]) => saveOutputBytes(...a),
    localGalleryList: (...a: unknown[]) => localGalleryList(...a),
  },
}));

import { useGenerationStore, suggestOutputFilename, needsHostRoute } from "./generation";
import { useConnectionStore } from "./connection";
import { useHostsStore } from "./hosts";

function request(): GenerateRequest {
  return { prompt: "a cat", model: "flux2-klein", width: 512, height: 512, steps: 4 };
}

const halRoute = {
  hostId: "hal9000-7680",
  label: "hal9000",
  kind: "remote" as const,
  target: { baseUrl: "http://hal9000:7680", apiKey: "hk" },
  // A print is admitted through the durable queue, so a frozen route must
  // carry the machine's instance identity and its advertised chunk limit.
  instanceId: "hal-instance",
  heterogeneousBatchMaxOutputs: 64,
  durableMedia: {
    protocol_version: 2,
    encrypted_at_rest: true,
    generate_request_media: true,
    identity: true,
    h3_references: true,
    private_h3: true,
  },
};

/** The target `admitGenerationBatch` was called against. */
function admittedTarget(call = 0): { baseUrl: string; apiKey: string | null } {
  return durableApi.admit.mock.calls[call]![0] as { baseUrl: string; apiKey: string | null };
}

/** The requests one durable admission carried. */
function admittedRequests(call = 0): GenerateRequest[] {
  return (durableApi.admit.mock.calls[call]![1] as { requests: GenerateRequest[] }).requests;
}

const chainDecision = {
  kind: "chain" as const,
  clipFrames: 97,
  motionTail: 17,
  stageCount: 3,
};

beforeEach(() => {
  setActivePinia(createPinia());
  primary.target = { baseUrl: "http://primary:7680", apiKey: "pk" };
  vi.clearAllMocks();
  apiFetchTo.mockResolvedValue(new Response(null, { status: 200 }));
  durableApi.admit.mockReset();
  durableApi.lookup.mockReset();
  durableApi.reconcile.mockReset();
  // Every print admits durably and stays queued unless a test says otherwise.
  durableApi.admit.mockImplementation(
    async (_target: unknown, body: { client_batch_id: string; requests: unknown[] }) => ({
      id: "server-batch",
      client_batch_id: body.client_batch_id,
      instance_id: "hal-instance",
      durable: true,
      children: body.requests.map((_request, index) => ({
        index: index + 1,
        job_id: `srv-${index + 1}`,
        state: "queued",
        created_at_ms: 1,
        updated_at_ms: 1,
      })),
    }),
  );
  streamableMediaUrl.mockResolvedValue("https://hal9000/media/generated-video");
  // Client ids restart with each fresh Pinia, so clear the module-scoped
  // per-job target map (a real session never reuses ids).
  useGenerationStore().resetJobs();
});

/** This device, ready and advertising the durable queue. */
function readyPrimary(): void {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://primary:7680", apiKey: "pk" };
  conn.status = "ready";
  const hosts = useHostsStore();
  hosts.telemetry.local = { instanceId: "local-instance" } as never;
  hosts.capabilities.local = {
    queue: { heterogeneous_batch_max_outputs: 64 },
    durable_media: {
      protocol_version: 2,
      encrypted_at_rest: true,
      generate_request_media: true,
      identity: true,
      h3_references: true,
      private_h3: true,
    },
  } as never;
}

/** One reachable remote machine, ready and advertising the durable queue. */
function readyRemote(): void {
  const hosts = useHostsStore();
  hosts.extras.push({
    id: "hal9000-7680",
    label: "hal9000",
    url: "http://hal9000:7680",
    apiKey: "hk",
    status: "ready",
    error: null,
    instanceId: "hal-instance",
  });
  hosts.telemetry["hal9000-7680"] = { instanceId: "hal-instance" } as never;
  hosts.capabilities["hal9000-7680"] = {
    queue: { heterogeneous_batch_max_outputs: 64 },
    durable_media: {
      protocol_version: 2,
      encrypted_at_rest: true,
      generate_request_media: true,
      identity: true,
      h3_references: true,
      private_h3: true,
    },
  } as never;
}

describe("generation store multi-host routing", () => {
  it("tags jobs with their host and admits against its target", async () => {
    const store = useGenerationStore();
    const { jobs } = store.submitBatch(request(), 1, halRoute);
    await flushPromises();
    expect(jobs[0]).toMatchObject({ hostId: "hal9000-7680", hostLabel: "hal9000" });
    expect(admittedTarget()).toEqual(halRoute.target);
    expect(sseStream).not.toHaveBeenCalled();
  });

  it("fails closed instead of falling back after a job target is released", async () => {
    const store = useGenerationStore();
    const { jobs } = store.submitBatch(request(), 1, halRoute);
    await flushPromises();

    expect(store.targetForJob(jobs[0]!.clientId)).toEqual(halRoute.target);
    store.resetJobs();
    expect(store.targetForJob(jobs[0]!.clientId)).toBeNull();
  });

  it("submits every per-item prompt through the one supplied route", async () => {
    const store = useGenerationStore();
    const prompts = ["first variation", "second variation", "third variation"];
    const { jobs } = store.submitBatch(request(), 3, halRoute, null, {
      prompts,
      originalPrompt: "source prompt",
    });
    await flushPromises();

    expect(jobs.map((job) => job.prompt)).toEqual(prompts);
    expect(durableApi.admit).toHaveBeenCalledTimes(1);
    expect(admittedTarget()).toEqual(halRoute.target);
    expect(admittedRequests().map((candidate) => candidate.original_prompt)).toEqual([
      "source prompt",
      "source prompt",
      "source prompt",
    ]);
  });

  it("rejects an inconsistent per-item prompt list before creating or streaming jobs", () => {
    const store = useGenerationStore();

    expect(() =>
      store.submitBatch(request(), 3, halRoute, null, {
        prompts: ["only one prompt"],
        originalPrompt: "source prompt",
      }),
    ).toThrow("Per-item prompt count 1 does not match batch size 3");
    expect(store.jobs).toHaveLength(0);
    expect(sseStream).not.toHaveBeenCalled();
  });

  it("admits an unrouted print against This device's own route", async () => {
    // An unrouted submit is This device: the app's embedded server is a
    // machine like any other, so its instance identity and advertised limit
    // are what the durable admission is frozen against.
    readyPrimary();
    const store = useGenerationStore();
    const { jobs } = store.submitBatch(request(), 1);
    await flushPromises();
    expect(jobs[0]?.hostId).toBe("local");
    expect(admittedTarget()).toEqual({ baseUrl: "http://primary:7680", apiKey: "pk" });
  });

  it("refuses an unrouted print when no machine can be resolved at all", async () => {
    const store = useGenerationStore();
    expect(() => store.submitBatch(request(), 1)).toThrow("No machine is selected for this print.");
    expect(store.jobs).toHaveLength(0);
    expect(durableApi.admit).not.toHaveBeenCalled();
  });

  it("falls back to a ready host when the primary connection is down", async () => {
    // Local engine failed to start (conn.info never set) but a remote host is
    // ready: an unrouted batch must be admitted against that host.
    readyRemote();
    const store = useGenerationStore();
    const { jobs } = store.submitBatch(request(), 1);
    await flushPromises();
    expect(jobs[0]).toMatchObject({ hostId: "hal9000-7680", hostLabel: "hal9000", remote: true });
    expect(admittedTarget()).toEqual({ baseUrl: "http://hal9000:7680", apiKey: "hk" });
  });

  it("keeps This device for unrouted jobs while the primary is ready", async () => {
    readyPrimary();
    readyRemote();
    const store = useGenerationStore();
    const { jobs } = store.submitBatch(request(), 1);
    await flushPromises();
    expect(jobs[0]?.hostId).toBe("local");
    expect(admittedTarget()).toEqual({ baseUrl: "http://primary:7680", apiKey: "pk" });
  });

  it("cancels a routed job against its own host", async () => {
    // Durable admission is what reports the queue id, and the cancel must
    // reach the machine that actually holds the job — matched by id AND by
    // the exact instance the job was admitted against.
    readyRemote();
    const store = useGenerationStore();
    const { jobs } = store.submitBatch(request(), 1, halRoute);
    await flushPromises();
    expect(jobs[0]!.id).toBe("srv-1");
    await store.cancel(jobs[0]!.clientId);
    const [target, path, init] = apiFetchTo.mock.calls[0] as [
      { baseUrl: string },
      string,
      { method: string },
    ];
    expect(target.baseUrl).toBe("http://hal9000:7680");
    expect(path).toBe("/api/queue/srv-1");
    expect(init.method).toBe("DELETE");
  });

  it("asks no host anything when no machine can be named for the print", async () => {
    // Never connected: the print reaches no machine, so it is refused by name
    // before a row exists — there is nothing to reconcile and no queue to ask
    // about, and every question would be about someone else's.
    primary.target = null;
    apiJsonTo.mockImplementation(() => Promise.resolve([]));
    const store = useGenerationStore();

    expect(() => store.submitBatch(request(), 1)).toThrow("No machine is selected for this print.");
    expect(apiJsonTo).not.toHaveBeenCalled();
    expect(durableApi.admit).not.toHaveBeenCalled();
    expect(store.jobs).toHaveLength(0);
  });

  it("posts only the supported auto-expand body and maps chain progress/completion", async () => {
    const createObjectUrl = vi.spyOn(URL, "createObjectURL");
    const chainRequest: GenerateRequest = {
      ...request(),
      model: "ltx-2.3-22b-distilled:fp8",
      prompt: "a lighthouse through a storm",
      negative_prompt: "text",
      width: 1536,
      height: 640,
      guidance: 3.5,
      seed: 42,
      output_format: "mp4",
      source_image: "source-b64",
      source_image_name: "source.png",
      strength: 0.7,
      frames: 241,
      fps: 24,
      enable_audio: true,
      loras: [{ path: "camera-control:dolly-in", scale: 1 }],
      pipeline: "distilled",
    };
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "progress",
          JSON.stringify({
            type: "chain_start",
            stage_count: 3,
            estimated_total_frames: 241,
            job_id: "chain-1",
          }),
        );
        opts.onEvent(
          "progress",
          JSON.stringify({
            type: "denoise_step",
            stage_idx: 1,
            step: 2,
            total: 4,
            job_id: "chain-1",
          }),
        );
        opts.onEvent(
          "complete",
          JSON.stringify({
            video: "aGVsbG8=",
            format: "mp4",
            width: 1536,
            height: 640,
            frames: 241,
            fps: 24,
            thumbnail: "thumb-b64",
            gif_preview: "gif-b64",
            has_audio: true,
            duration_ms: 10_042,
            stage_count: 3,
            generation_time_ms: 12_345,
            filename: "chain-42.mp4",
            metadata: {
              prompt: chainRequest.prompt,
              model: chainRequest.model,
              seed: 42,
              steps: 4,
              guidance: 3.5,
              width: 1536,
              height: 640,
            },
            script: {},
          }),
        );
        return Promise.resolve();
      },
    );

    const { jobs, settled } = useGenerationStore().submitBatch(
      chainRequest,
      1,
      { ...halRoute, mirrorRemoteOutput: false },
      chainDecision,
    );
    await settled;

    const [path, options] = sseStream.mock.calls[0] as [
      string,
      { body: Record<string, unknown>; headers?: Record<string, string> },
    ];
    expect(path).toBe("/api/generate/chain/stream");
    expect(options.body).toEqual({
      output_mode: "one-shot",
      model: chainRequest.model,
      prompt: chainRequest.prompt,
      total_frames: 241,
      clip_frames: 97,
      motion_tail_frames: 17,
      width: 1536,
      height: 640,
      fps: 24,
      seed: 42,
      steps: 4,
      guidance: 3.5,
      strength: 0.7,
      output_format: "mp4",
      source_image: "source-b64",
      enable_audio: true,
    });
    expect(options.body).not.toHaveProperty("negative_prompt");
    expect(options.body).not.toHaveProperty("loras");
    expect(options.body).not.toHaveProperty("pipeline");
    expect(options.headers).toBeUndefined();
    expect(jobs[0]).toMatchObject({
      id: "chain-1",
      status: "complete",
      chainStageCount: 3,
      resultUrlIsObjectUrl: true,
      result: {
        image: "aGVsbG8=",
        filename: "chain-42.mp4",
        seed_used: 42,
        video_frames: 241,
        video_fps: 24,
        video_has_audio: true,
      },
    });
    expect(createObjectUrl).toHaveBeenCalledTimes(1);
  });

  it("keeps a metadata-only chain filename without creating a media Blob", async () => {
    const createObjectUrl = vi.spyOn(URL, "createObjectURL");
    streamableMediaUrl.mockResolvedValueOnce("https://hal9000/media/chain-video");
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "progress",
          JSON.stringify({
            type: "chain_start",
            stage_count: 3,
            estimated_total_frames: 241,
            job_id: "chain-metadata",
          }),
        );
        opts.onEvent(
          "complete",
          JSON.stringify({
            video: "",
            format: "mp4",
            width: 1536,
            height: 640,
            frames: 241,
            fps: 24,
            stage_count: 3,
            generation_time_ms: 12_345,
            filename: "metadata chain.mp4",
            metadata: {
              prompt: "a lighthouse",
              model: "ltx-2.3-22b-distilled:fp8",
              seed: 77,
              steps: 4,
              guidance: 3.5,
              width: 1536,
              height: 640,
            },
            script: {},
          }),
        );
        return Promise.resolve();
      },
    );
    const chainRequest: GenerateRequest = {
      ...request(),
      model: "ltx-2.3-22b-distilled:fp8",
      width: 1536,
      height: 640,
      guidance: 3.5,
      frames: 241,
      fps: 24,
      output_format: "mp4",
    };

    const { jobs, settled } = useGenerationStore().submitBatch(
      chainRequest,
      1,
      {
        ...halRoute,
        mirrorRemoteOutput: false,
        retainEncodedResult: false,
        metadataOnlyCompletion: true,
      },
      chainDecision,
    );
    await settled;
    await vi.waitFor(() => expect(jobs[0]!.resultUrl).toBe("https://hal9000/media/chain-video"));

    expect(sseStream.mock.calls[0]?.[0]).toBe("/api/generate/chain/stream");
    expect(sseStream.mock.calls[0]?.[1]).toMatchObject({
      headers: { "X-Mold-SSE-Payload": "metadata-only" },
    });
    expect(streamableMediaUrl).toHaveBeenCalledWith("/api/gallery/image/metadata%20chain.mp4", {
      target: halRoute.target,
      cacheKey: halRoute.hostId,
      allowLegacyBlob: false,
    });
    expect(createObjectUrl).not.toHaveBeenCalled();
    expect(jobs[0]!.result).toMatchObject({
      image: "",
      filename: "metadata chain.mp4",
      seed_used: 77,
      video_frames: 241,
    });
  });

  it("falls back to the encoded chain video when an older host ignores metadata-only", async () => {
    const createObjectUrl = vi.spyOn(URL, "createObjectURL");
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "complete",
          JSON.stringify({
            video: "aGVsbG8=",
            format: "mp4",
            width: 1536,
            height: 640,
            frames: 241,
            fps: 24,
            stage_count: 3,
            generation_time_ms: 12_345,
            script: {},
          }),
        );
        return Promise.resolve();
      },
    );
    const chainRequest: GenerateRequest = {
      ...request(),
      model: "ltx-2.3-22b-distilled:fp8",
      width: 1536,
      height: 640,
      frames: 241,
      fps: 24,
      output_format: "mp4",
    };

    const { jobs, settled } = useGenerationStore().submitBatch(
      chainRequest,
      1,
      {
        ...halRoute,
        mirrorRemoteOutput: false,
        retainEncodedResult: false,
        metadataOnlyCompletion: true,
      },
      chainDecision,
    );
    await settled;

    expect(sseStream.mock.calls[0]?.[1]).toMatchObject({
      headers: { "X-Mold-SSE-Payload": "metadata-only" },
    });
    expect(streamableMediaUrl).not.toHaveBeenCalled();
    expect(createObjectUrl).toHaveBeenCalledTimes(1);
    expect(jobs[0]).toMatchObject({
      status: "complete",
      resultUrlIsObjectUrl: true,
      result: { image: "", format: "mp4", video_frames: 241 },
    });
  });

  it("cancels an automatic chain through the durable chain-job endpoint", async () => {
    sseStream.mockImplementation(
      (_path: string, opts: { signal: AbortSignal; onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "progress",
          JSON.stringify({
            type: "chain_start",
            stage_count: 3,
            estimated_total_frames: 241,
            job_id: "chain/job-1",
          }),
        );
        return new Promise<void>((resolve) => {
          opts.signal.addEventListener("abort", () => resolve());
        });
      },
    );
    const store = useGenerationStore();
    const { jobs } = store.submitBatch(
      { ...request(), model: "ltx-2.3-22b-distilled:fp8", frames: 241, fps: 24 },
      1,
      halRoute,
      chainDecision,
    );
    await Promise.resolve();
    await store.cancel(jobs[0]!.clientId);

    expect(apiFetchTo).toHaveBeenCalledWith(
      halRoute.target,
      "/api/chain-jobs/chain%2Fjob-1/cancel",
      { method: "POST" },
    );
    expect(jobs[0]).toMatchObject({ status: "error", error: "Cancelled" });
  });

  it("rejects a defensive reject decision before creating jobs", () => {
    const store = useGenerationStore();
    expect(() =>
      store.submitBatch(request(), 1, halRoute, {
        kind: "reject",
        reason: "This model cannot chain.",
      }),
    ).toThrow("This model cannot chain.");
    expect(store.jobs).toHaveLength(0);
  });
});

describe("needsHostRoute", () => {
  it("routes with multiple hosts, or a dead primary plus a live host", () => {
    // Multi-host always routes (existing behavior).
    expect(needsHostRoute({ multiHost: true, primaryReady: true, anyHostReady: true })).toBe(true);
    // Single ready primary: unrouted (existing behavior).
    expect(needsHostRoute({ multiHost: false, primaryReady: true, anyHostReady: true })).toBe(
      false,
    );
    // Dead primary but a ready host exists: must route to reach it.
    expect(needsHostRoute({ multiHost: false, primaryReady: false, anyHostReady: true })).toBe(
      true,
    );
    // Nothing is ready: unrouted, so the submit surfaces the directed error.
    expect(needsHostRoute({ multiHost: false, primaryReady: false, anyHostReady: false })).toBe(
      false,
    );
  });
});

describe("suggestOutputFilename", () => {
  it("builds a filesystem-safe name from model, seed, and format", () => {
    expect(suggestOutputFilename("flux-dev:q8", 42, "png", 1700000000000)).toBe(
      "mold-flux-dev-q8-42-1700000000000.png",
    );
    expect(suggestOutputFilename("hf:Qwen/Qwen-Image", 1, "webp", 2)).toBe(
      "mold-hf-qwen-qwen-image-1-2.webp",
    );
  });
});
