import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import type { GenerateRequest } from "../lib/api/types";

const sseStream = vi.fn();
vi.mock("../lib/api/sse", () => ({
  sseStream: (...a: unknown[]) => sseStream(...a),
}));

const apiFetchTo = vi.fn().mockResolvedValue(new Response(null, { status: 200 }));
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
  currentTarget: () => ({ baseUrl: "http://primary:7680", apiKey: "pk" }),
}));

vi.mock("../lib/notify", () => ({
  notifyGenerated: vi.fn(),
  notifyGenerationFailed: vi.fn(),
}));

const saveOutputBytes = vi.fn().mockResolvedValue("saved.png");
vi.mock("../lib/ipc", () => ({
  inTauri: () => true,
  ipc: {
    saveOutputBytes: (...a: unknown[]) => saveOutputBytes(...a),
  },
}));

import { useGenerationStore, suggestOutputFilename } from "./generation";
import { useAppPrefsStore } from "./appPrefs";

function request(): GenerateRequest {
  return { prompt: "a cat", model: "flux2-klein", width: 512, height: 512, steps: 4 };
}

const halRoute = {
  hostId: "hal9000-7680",
  label: "hal9000",
  kind: "remote" as const,
  target: { baseUrl: "http://hal9000:7680", apiKey: "hk" },
};

function completeFrame() {
  return JSON.stringify({
    image: "aGVsbG8=",
    format: "png",
    width: 512,
    height: 512,
    seed_used: 7,
    generation_time_ms: 100,
    model: "flux2-klein",
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  apiFetchTo.mockResolvedValue(new Response(null, { status: 200 }));
  // Client ids restart with each fresh Pinia, so clear the module-scoped
  // per-job target map (a real session never reuses ids).
  useGenerationStore().resetJobs();
});

describe("generation store multi-host routing", () => {
  it("tags jobs with their host and streams against its target", async () => {
    sseStream.mockResolvedValue(undefined);
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(request(), 1, halRoute);
    await settled;
    expect(jobs[0]).toMatchObject({ hostId: "hal9000-7680", hostLabel: "hal9000" });
    const options = sseStream.mock.calls[0]?.[1] as { target?: { baseUrl: string } };
    expect(options.target?.baseUrl).toBe("http://hal9000:7680");
  });

  it("falls back to the primary connection when no route is given", async () => {
    sseStream.mockResolvedValue(undefined);
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(request(), 1);
    await settled;
    expect(jobs[0]?.hostId).toBeNull();
    const options = sseStream.mock.calls[0]?.[1] as { target?: unknown };
    expect(options.target).toBeUndefined();
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
    expect(saveOutputBytes).toHaveBeenCalledTimes(1);
    const [filename, b64] = saveOutputBytes.mock.calls[0] as [string, string];
    expect(filename).toMatch(/^mold-flux2-klein-7-\d+\.png$/);
    expect(b64).toBe("aGVsbG8=");
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
