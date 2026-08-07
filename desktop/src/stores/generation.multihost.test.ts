import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import type { GenerateRequest } from "../lib/api/types";

const sseStream = vi.fn();
vi.mock("../lib/api/sse", () => ({
  sseStream: (...a: unknown[]) => sseStream(...a),
}));

const prepareReferenceUploads = vi.fn();
vi.mock("@studio/api/referenceUploads", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/referenceUploads")>()),
  prepareReferenceUploads: (...args: unknown[]) => prepareReferenceUploads(...args),
}));

const streamableMediaUrl = vi.fn().mockResolvedValue("https://hal9000/media/generated-video");
const evictMedia = vi.fn();
vi.mock("../lib/gallery/media", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/gallery/media")>()),
  streamableMediaUrl: (...a: unknown[]) => streamableMediaUrl(...a),
  evictMedia: (...a: unknown[]) => evictMedia(...a),
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
  currentTarget: () => ({ baseUrl: "http://primary:7680", apiKey: "pk" }),
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
import { useAppPrefsStore } from "./appPrefs";
import { useConnectionStore } from "./connection";
import { useGalleryStore } from "./gallery";
import { useHostsStore } from "./hosts";

function request(): GenerateRequest {
  return { prompt: "a cat", model: "flux2-klein", width: 512, height: 512, steps: 4 };
}

const halRoute = {
  hostId: "hal9000-7680",
  label: "hal9000",
  kind: "remote" as const,
  target: { baseUrl: "http://hal9000:7680", apiKey: "hk" },
};

const chainDecision = {
  kind: "chain" as const,
  clipFrames: 97,
  motionTail: 17,
  stageCount: 3,
};

function completeFrame() {
  return JSON.stringify({
    image: "aGVsbG8=",
    original_image: "b3JpZ2luYWw=",
    format: "png",
    width: 512,
    height: 512,
    original_width: 128,
    original_height: 128,
    seed_used: 7,
    generation_time_ms: 100,
    model: "flux2-klein",
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  apiFetchTo.mockResolvedValue(new Response(null, { status: 200 }));
  streamableMediaUrl.mockResolvedValue("https://hal9000/media/generated-video");
  // Client ids restart with each fresh Pinia, so clear the module-scoped
  // per-job target map (a real session never reuses ids).
  useGenerationStore().resetJobs();
});

describe("generation store multi-host routing", () => {
  it("creates a fresh exact-host reference lease for every submission attempt", async () => {
    let attempt = 0;
    const cancel = vi.fn().mockResolvedValue(undefined);
    prepareReferenceUploads.mockImplementation(
      async ({ request: original }: { request: GenerateRequest }) => {
        attempt += 1;
        return {
          request: {
            ...original,
            references: original.references?.map((reference) => ({
              ...reference,
              media: { authority: "upload", handle: `lease-${attempt}` },
            })),
          },
          expiresAtMs: Date.now() + 60_000,
          requestScopeSha256: "a".repeat(64),
          cancel,
        };
      },
    );
    sseStream.mockResolvedValue(undefined);
    const original: GenerateRequest = {
      ...request(),
      model: "minimax-h3-ref2va",
      frames: 124,
      fps: 24,
      references: [
        {
          kind: "image",
          media: { authority: "inline", data: "PRIVATE-IMAGE-BYTES" },
          provenance: { name: "identity.png", sha256: "b".repeat(64) },
          mime_type: "image/png",
          width: 32,
          height: 24,
        },
      ],
    };
    const route = {
      ...halRoute,
      instanceId: "instance-1",
      referenceUploads: {
        available: true,
        protocol_version: 1,
        requires_api_key: true,
        session_path: "/api/generate/reference-upload-sessions",
        upload_path: "/api/generate/reference-upload",
        session_handle_header: "x-mold-reference-session",
        upload_handle_header: "x-mold-reference-upload",
        max_file_bytes: 1_000_000,
        max_session_bytes: 2_000_000,
        session_ttl_ms: 60_000,
      },
    };

    const first = useGenerationStore().submitBatch(original, 1, route);
    await first.settled;
    const second = useGenerationStore().submitBatch(original, 1, route);
    await second.settled;

    expect(prepareReferenceUploads).toHaveBeenCalledTimes(2);
    expect(prepareReferenceUploads.mock.calls[0]?.[0]).toMatchObject({
      target: halRoute.target,
      expectedInstanceId: "instance-1",
      capabilities: route.referenceUploads,
    });
    expect(sseStream.mock.calls[0]?.[1].body.references[0].media).toEqual({
      authority: "upload",
      handle: "lease-1",
    });
    expect(sseStream.mock.calls[1]?.[1].body.references[0].media).toEqual({
      authority: "upload",
      handle: "lease-2",
    });
    expect(first.jobs[0]!.request!.references?.[0]?.media).toEqual({
      authority: "inline",
      data: "PRIVATE-IMAGE-BYTES",
    });
    expect(second.jobs[0]!.request!.references?.[0]?.media).toEqual({
      authority: "inline",
      data: "PRIVATE-IMAGE-BYTES",
    });
    expect(cancel).toHaveBeenCalledTimes(2);
  });

  it("tags jobs with their host and streams against its target", async () => {
    sseStream.mockResolvedValue(undefined);
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(request(), 1, halRoute);
    await settled;
    expect(jobs[0]).toMatchObject({ hostId: "hal9000-7680", hostLabel: "hal9000" });
    const options = sseStream.mock.calls[0]?.[1] as { target?: { baseUrl: string } };
    expect(options.target?.baseUrl).toBe("http://hal9000:7680");
  });

  it("fails closed instead of falling back after a job target is released", async () => {
    sseStream.mockResolvedValue(undefined);
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(request(), 1, halRoute);
    await settled;

    expect(store.targetForJob(jobs[0]!.clientId)).toEqual(halRoute.target);
    store.resetJobs();
    expect(store.targetForJob(jobs[0]!.clientId)).toBeNull();
  });

  it("submits every per-item prompt through the one supplied route", async () => {
    sseStream.mockResolvedValue(undefined);
    const store = useGenerationStore();
    const prompts = ["first variation", "second variation", "third variation"];
    const { jobs, settled } = store.submitBatch(request(), 3, halRoute, null, {
      prompts,
      originalPrompt: "source prompt",
    });
    await settled;

    expect(jobs.map((job) => job.prompt)).toEqual(prompts);
    expect(sseStream).toHaveBeenCalledTimes(3);
    for (const [, options] of sseStream.mock.calls) {
      expect(options).toMatchObject({ target: halRoute.target });
    }
    expect(
      sseStream.mock.calls.map(
        ([, options]) => (options as { body: GenerateRequest }).body.original_prompt,
      ),
    ).toEqual(["source prompt", "source prompt", "source prompt"]);
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

  it("snapshots the primary target at submit when no route is given", async () => {
    sseStream.mockResolvedValue(undefined);
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(request(), 1);
    await settled;
    expect(jobs[0]?.hostId).toBeNull();
    // The target is pinned at submit time — a mid-batch primary switch must
    // not reroute queued siblings or cancels to the new engine.
    const options = sseStream.mock.calls[0]?.[1] as { target?: { baseUrl: string } };
    expect(options.target?.baseUrl).toBe("http://primary:7680");
  });

  it("auto-saves remote results to this Mac when the pref is on", async () => {
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent("complete", completeFrame());
        return Promise.resolve();
      },
    );
    const store = useGenerationStore();
    const { settled } = store.submitBatch(request(), 1, halRoute);
    await settled;
    expect(saveOutputBytes).toHaveBeenCalledTimes(2);
    const [originalName, originalB64] = saveOutputBytes.mock.calls[0] as [string, string];
    const [upscaledName, upscaledB64] = saveOutputBytes.mock.calls[1] as [string, string];
    expect(originalName).toMatch(/^mold-flux2-klein-7-\d+-original\.png$/);
    expect(originalB64).toBe("b3JpZ2luYWw=");
    expect(upscaledName).toMatch(/^mold-flux2-klein-7-\d+-upscaled\.png$/);
    expect(upscaledB64).toBe("aGVsbG8=");
  });

  it("keeps the server's gallery filenames and metadata on auto-saved copies", async () => {
    const metadata = {
      prompt: "a cheetah at dusk",
      model: "flux2-klein",
      seed: 7,
      steps: 28,
      guidance: 3.5,
      width: 512,
      height: 512,
    };
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "complete",
          JSON.stringify({
            ...JSON.parse(completeFrame()),
            filename: "flux2-klein-999-upscaled.png",
            original_filename: "flux2-klein-999-original.png",
            metadata,
          }),
        );
        return Promise.resolve();
      },
    );
    const store = useGenerationStore();
    const { settled } = store.submitBatch(request(), 1, halRoute);
    await settled;

    expect(saveOutputBytes).toHaveBeenCalledTimes(2);
    const [originalName, , originalMeta] = saveOutputBytes.mock.calls[0] as [
      string,
      string,
      Record<string, unknown>,
    ];
    const [upscaledName, , upscaledMeta] = saveOutputBytes.mock.calls[1] as [
      string,
      string,
      Record<string, unknown>,
    ];
    // The origin's names are kept verbatim — the copy and the original stay
    // one logical print in the merged gallery.
    expect(originalName).toBe("flux2-klein-999-original.png");
    expect(upscaledName).toBe("flux2-klein-999-upscaled.png");
    // The recorded metadata rides along; the original gets its true dims.
    expect(upscaledMeta).toMatchObject({ seed: 7, width: 512, height: 512 });
    expect(originalMeta).toMatchObject({ seed: 7, width: 128, height: 128 });
  });

  it("refreshes the local gallery when only one paired remote save succeeds", async () => {
    saveOutputBytes
      .mockRejectedValueOnce(new Error("original save failed"))
      .mockResolvedValueOnce("upscaled.png");
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent("complete", completeFrame());
        return Promise.resolve();
      },
    );
    useGalleryStore().buckets["local"] = { items: [], loading: false, error: null, loaded: true };

    await useGenerationStore().submitBatch(request(), 1, halRoute).settled;

    await vi.waitFor(() => expect(localGalleryList).toHaveBeenCalled());
  });

  it("skips the local save for local jobs and when the pref is off", async () => {
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent("complete", completeFrame());
        return Promise.resolve();
      },
    );
    const store = useGenerationStore();
    // Local (unrouted) job: never saved — it's already in the local gallery.
    await store.submitBatch(request(), 1).settled;
    expect(saveOutputBytes).not.toHaveBeenCalled();
    // Remote job with the pref off: not saved either.
    useAppPrefsStore().settings = { saveRemoteOutputs: false } as never;
    await store.submitBatch(request(), 1, halRoute).settled;
    expect(saveOutputBytes).not.toHaveBeenCalled();
  });

  it("keeps an iPhone remote job remote without mirroring into a desktop gallery", async () => {
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "complete",
          JSON.stringify({
            ...JSON.parse(completeFrame()),
            video_thumbnail: "large-thumbnail-base64",
            video_gif_preview: "large-gif-base64",
          }),
        );
        return Promise.resolve();
      },
    );
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(request(), 1, {
      ...halRoute,
      mirrorRemoteOutput: false,
      retainEncodedResult: false,
    });
    await settled;

    expect(jobs[0]).toMatchObject({
      remote: true,
      mirrorRemoteOutput: false,
      retainEncodedResult: false,
      result: {
        image: "",
        original_image: null,
        video_thumbnail: null,
        video_gif_preview: null,
        seed_used: 7,
      },
    });
    expect(saveOutputBytes).not.toHaveBeenCalled();
  });

  it("loads an iPhone video from its saved host file without decoding SSE media", async () => {
    const createObjectUrl = vi.spyOn(URL, "createObjectURL");
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "complete",
          JSON.stringify({
            ...JSON.parse(completeFrame()),
            image: "",
            format: "mp4",
            filename: "generated clip.mp4",
            original_image: null,
            video_frames: 121,
            video_fps: 30,
          }),
        );
        return Promise.resolve();
      },
    );
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(request(), 1, {
      ...halRoute,
      mirrorRemoteOutput: false,
      retainEncodedResult: false,
      metadataOnlyCompletion: true,
    });
    await settled;
    await vi.waitFor(() =>
      expect(jobs[0]!.resultUrl).toBe("https://hal9000/media/generated-video"),
    );

    expect(streamableMediaUrl).toHaveBeenCalledWith("/api/gallery/image/generated%20clip.mp4", {
      target: halRoute.target,
      cacheKey: halRoute.hostId,
      allowLegacyBlob: false,
    });
    expect(sseStream.mock.calls[0]?.[1]).toMatchObject({
      headers: { "X-Mold-SSE-Payload": "metadata-only" },
    });
    expect(createObjectUrl).not.toHaveBeenCalled();
    expect(jobs[0]!.result?.image).toBe("");
  });

  it("does not mirror a metadata-only completion without encoded media", async () => {
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "complete",
          JSON.stringify({
            ...JSON.parse(completeFrame()),
            image: "",
            original_image: null,
            filename: "generated image.png",
          }),
        );
        return Promise.resolve();
      },
    );

    await useGenerationStore().submitBatch(request(), 1, {
      ...halRoute,
      metadataOnlyCompletion: true,
    }).settled;

    expect(saveOutputBytes).not.toHaveBeenCalled();
  });

  it("loads a metadata-only iPhone image from its saved host file", async () => {
    const createObjectUrl = vi.spyOn(URL, "createObjectURL");
    streamableMediaUrl.mockResolvedValueOnce("https://hal9000/media/generated-image");
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "complete",
          JSON.stringify({
            ...JSON.parse(completeFrame()),
            image: "",
            original_image: null,
            filename: "generated image.png",
          }),
        );
        return Promise.resolve();
      },
    );

    const { jobs, settled } = useGenerationStore().submitBatch(request(), 1, {
      ...halRoute,
      mirrorRemoteOutput: false,
      retainEncodedResult: false,
      metadataOnlyCompletion: true,
    });
    await settled;
    await vi.waitFor(() =>
      expect(jobs[0]!.resultUrl).toBe("https://hal9000/media/generated-image"),
    );

    expect(streamableMediaUrl).toHaveBeenCalledWith("/api/gallery/image/generated%20image.png", {
      target: halRoute.target,
      cacheKey: halRoute.hostId,
      allowLegacyBlob: true,
    });
    expect(createObjectUrl).not.toHaveBeenCalled();
  });

  it("surfaces a metadata-only completion that has no saved filename", async () => {
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "complete",
          JSON.stringify({ ...JSON.parse(completeFrame()), image: "", original_image: null }),
        );
        return Promise.resolve();
      },
    );

    const { jobs, settled } = useGenerationStore().submitBatch(request(), 1, {
      ...halRoute,
      mirrorRemoteOutput: false,
      retainEncodedResult: false,
      metadataOnlyCompletion: true,
    });
    await settled;
    await vi.waitFor(() => expect(jobs[0]!.resultError).toContain("saved result URL"));

    expect(jobs[0]).toMatchObject({ status: "complete", resultUrl: null });
    expect(streamableMediaUrl).not.toHaveBeenCalled();
  });

  it("renews a ticketed result URL when it is close to expiring", async () => {
    const now = vi.spyOn(Date, "now").mockReturnValue(1_800_000_000_000);
    streamableMediaUrl
      .mockResolvedValueOnce(
        "https://hal9000/media/generated-video?media_token=old&expires=1800000300",
      )
      .mockResolvedValueOnce(
        "https://hal9000/media/generated-video?media_token=new&expires=1800001200",
      );
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "complete",
          JSON.stringify({
            ...JSON.parse(completeFrame()),
            image: "",
            format: "mp4",
            filename: "generated clip.mp4",
            original_image: null,
          }),
        );
        return Promise.resolve();
      },
    );
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(request(), 1, {
      ...halRoute,
      mirrorRemoteOutput: false,
      retainEncodedResult: false,
      metadataOnlyCompletion: true,
    });
    await settled;
    await vi.waitFor(() => expect(jobs[0]!.resultUrl).toContain("media_token=old"));

    await store.refreshRemoteResultUrl(jobs[0]!.clientId);
    expect(streamableMediaUrl).toHaveBeenCalledTimes(1);
    now.mockReturnValue(1_800_000_250_000);
    await store.refreshRemoteResultUrl(jobs[0]!.clientId);

    expect(streamableMediaUrl).toHaveBeenCalledTimes(2);
    expect(jobs[0]!.resultUrl).toContain("media_token=new");
    now.mockRestore();
  });

  it("evicts a legacy generated-image Blob before a forced retry", async () => {
    const revokeObjectUrl = vi.spyOn(URL, "revokeObjectURL");
    streamableMediaUrl
      .mockResolvedValueOnce("blob:legacy-generated-image")
      .mockResolvedValueOnce("blob:refetched-generated-image");
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent(
          "complete",
          JSON.stringify({
            ...JSON.parse(completeFrame()),
            image: "",
            filename: "generated image.png",
            original_image: null,
          }),
        );
        return Promise.resolve();
      },
    );
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(request(), 1, {
      ...halRoute,
      mirrorRemoteOutput: false,
      retainEncodedResult: false,
      metadataOnlyCompletion: true,
    });
    await settled;
    await vi.waitFor(() => expect(jobs[0]!.resultUrl).toBe("blob:legacy-generated-image"));

    await store.refreshRemoteResultUrl(jobs[0]!.clientId, true);

    expect(evictMedia).toHaveBeenCalledWith(
      "/api/gallery/image/generated%20image.png",
      halRoute.hostId,
    );
    expect(evictMedia.mock.invocationCallOrder[0]).toBeLessThan(
      streamableMediaUrl.mock.invocationCallOrder[1]!,
    );
    expect(jobs[0]!.resultUrl).toBe("blob:refetched-generated-image");
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:legacy-generated-image");
  });

  it("refreshes the origin host's loaded gallery bucket when a routed job completes", async () => {
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent("complete", completeFrame());
        return Promise.resolve();
      },
    );
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: "hk",
      status: "ready",
      error: null,
      instanceId: null,
    });
    const gallery = useGalleryStore();
    gallery.buckets["hal9000-7680"] = { items: [], loading: false, error: null, loaded: true };
    // The auto local save also refreshes this Mac's loaded bucket.
    gallery.buckets["local"] = { items: [], loading: false, error: null, loaded: true };

    const store = useGenerationStore();
    await store.submitBatch(request(), 1, halRoute).settled;

    await vi.waitFor(() => {
      expect(apiJsonTo).toHaveBeenCalledWith(
        { baseUrl: "http://hal9000:7680", apiKey: "hk" },
        "/api/gallery",
      );
      expect(localGalleryList).toHaveBeenCalled();
    });
  });

  it("never force-loads gallery buckets from a background completion", async () => {
    sseStream.mockImplementation(
      (_path: string, opts: { onEvent: (e: string, d: string) => void }) => {
        opts.onEvent("complete", completeFrame());
        return Promise.resolve();
      },
    );
    const store = useGenerationStore();
    await store.submitBatch(request(), 1, halRoute).settled;
    await Promise.resolve();
    expect(apiJsonTo).not.toHaveBeenCalled();
    expect(useGalleryStore().buckets["hal9000-7680"]).toBeUndefined();
  });

  it("falls back to a ready host when the primary connection is down", async () => {
    // Local engine failed to start (conn.info never set) but a remote host is
    // ready: an unrouted batch must snapshot that host, not the dead primary.
    sseStream.mockResolvedValue(undefined);
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: "hk",
      status: "ready",
      error: null,
      instanceId: null,
    });
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(request(), 1);
    await settled;
    expect(jobs[0]).toMatchObject({ hostId: "hal9000-7680", hostLabel: "hal9000", remote: true });
    const options = sseStream.mock.calls[0]?.[1] as { target?: { baseUrl: string } };
    expect(options.target?.baseUrl).toBe("http://hal9000:7680");
  });

  it("keeps the primary snapshot for unrouted jobs while the primary is ready", async () => {
    sseStream.mockResolvedValue(undefined);
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://primary:7680", apiKey: "pk" };
    conn.status = "ready";
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: "hk",
      status: "ready",
      error: null,
      instanceId: null,
    });
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(request(), 1);
    await settled;
    expect(jobs[0]?.hostId).toBeNull();
    const options = sseStream.mock.calls[0]?.[1] as { target?: { baseUrl: string } };
    expect(options.target?.baseUrl).toBe("http://primary:7680");
  });

  it("cancels a routed job against its own host", async () => {
    // Stream that stays open until aborted, reporting the server id.
    sseStream.mockImplementation(
      (_path: string, opts: { signal: AbortSignal; onEvent: (e: string, d: string) => void }) => {
        opts.onEvent("progress", JSON.stringify({ type: "queued", position: 1, id: "srv-1" }));
        return new Promise<void>((resolve) => {
          opts.signal.addEventListener("abort", () => resolve());
        });
      },
    );
    const store = useGenerationStore();
    const { jobs } = store.submitBatch(request(), 1, halRoute);
    await Promise.resolve(); // let the stream open and deliver "queued"
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
