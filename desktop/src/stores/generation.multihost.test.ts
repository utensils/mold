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

import { useGenerationStore } from "./generation";

function request(): GenerateRequest {
  return { prompt: "a cat", model: "flux2-klein", width: 512, height: 512, steps: 4 };
}

const halRoute = {
  hostId: "hal9000-7680",
  label: "hal9000",
  target: { baseUrl: "http://hal9000:7680", apiKey: "hk" },
};

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
