import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GalleryImage, GenerateRequest } from "../lib/api/types";
import { newJob, type Job } from "../lib/generationJob";

const { apiFetchTo, apiJsonTo } = vi.hoisted(() => ({
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(),
}));

vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiFetchTo,
  apiJsonTo,
}));

import { ApiError } from "../lib/api/client";
import {
  galleryCompletion,
  isInterruptedGenerationError,
  matchGalleryPrint,
  reconcileInterruptedGenerationJobs,
} from "./mobileGenerationRecovery";

const target = { baseUrl: "http://studio.tailnet.ts.net:7680", apiKey: "secret" };

function makeJob(overrides: Partial<Job> = {}): Job {
  const job = newJob({
    prompt: "a ship crossing violet lightning",
    model: "ltx2:q8",
    width: 768,
    height: 512,
    steps: 28,
    seed: 77,
  } as GenerateRequest);
  job.clientId = 7;
  job.batchId = 1;
  job.id = "job-9";
  job.hostId = "studio-id";
  job.hostLabel = "Studio";
  job.remote = true;
  job.metadataOnlyCompletion = true;
  job.streamStarted = true;
  job.status = "error";
  job.error = "Load failed";
  return Object.assign(job, overrides);
}

const galleryPrint: GalleryImage = {
  filename: "resumed print.png",
  timestamp: 1_700_000_000,
  format: "png",
  metadata: {
    prompt: "a ship crossing violet lightning",
    model: "ltx2:q8",
    seed: 77,
    steps: 28,
    guidance: 4,
    width: 768,
    height: 512,
  },
};

function options(overrides: Record<string, unknown> = {}) {
  return {
    target,
    hostLabel: "Studio",
    refreshResultUrl: vi.fn(),
    pollIntervalMs: 0,
    sleep: () => Promise.resolve(),
    ...overrides,
  };
}

beforeEach(() => {
  apiFetchTo.mockReset().mockResolvedValue(new Response(null, { status: 204 }));
  apiJsonTo.mockReset();
});

describe("isInterruptedGenerationError", () => {
  it("recognizes dead-socket errors and the empty-close fallback", () => {
    expect(isInterruptedGenerationError("Load failed")).toBe(true);
    expect(isInterruptedGenerationError("The network connection was lost.")).toBe(true);
    expect(isInterruptedGenerationError("The generation stream closed before completion.")).toBe(
      true,
    );
  });

  it("never reconciles server-authored failures or cancellations", () => {
    expect(isInterruptedGenerationError("host ran out of memory")).toBe(false);
    expect(isInterruptedGenerationError("Cancelled")).toBe(false);
    expect(
      isInterruptedGenerationError("Cancelled locally; remote cancellation was not confirmed."),
    ).toBe(false);
    expect(isInterruptedGenerationError(null)).toBe(false);
  });
});

describe("matchGalleryPrint", () => {
  it("joins on the explicit seed and model", () => {
    const job = makeJob();
    expect(matchGalleryPrint(job, [galleryPrint])).toBe(galleryPrint);
    expect(matchGalleryPrint(makeJob({ visualSeed: "78" }), [galleryPrint])).toBeNull();
    expect(matchGalleryPrint(makeJob({ model: "flux:q8" }), [galleryPrint])).toBeNull();
  });

  it("uses the recorded prompt as the prepared-sibling tiebreaker", () => {
    expect(
      matchGalleryPrint(makeJob({ prompt: "a different sibling" }), [galleryPrint]),
    ).toBeNull();
  });

  it("rejects a stale same-seed print whose dimensions or steps differ", () => {
    // A fixed-seed re-run at a new resolution must not resurrect yesterday's
    // print: the join requires dims and steps to agree when both sides know
    // them.
    expect(matchGalleryPrint(makeJob({ width: 1024, height: 1024 }), [galleryPrint])).toBeNull();
    expect(matchGalleryPrint(makeJob({ total: 8 }), [galleryPrint])).toBeNull();
    // Absent metadata on either side stays permissive — old hosts omit steps.
    const { steps: _s, width: _w, height: _h, ...bareMeta } = galleryPrint.metadata;
    const bare = { ...galleryPrint, metadata: bareMeta } as GalleryImage;
    expect(matchGalleryPrint(makeJob({ width: 1024, height: 1024 }), [bare])).toBe(bare);
  });

  it("requires an exact numeric seed", () => {
    expect(matchGalleryPrint(makeJob({ visualSeed: "ltx2:q8·prompt" }), [galleryPrint])).toBeNull();
  });
});

describe("galleryCompletion", () => {
  it("synthesizes the metadata-only completion the lost frame would have carried", () => {
    const complete = galleryCompletion(galleryPrint);
    expect(complete).toMatchObject({
      image: "",
      format: "png",
      filename: "resumed print.png",
      seed_used: 77,
      model: "ltx2:q8",
      generation_time_ms: 0,
    });
    expect(complete.metadata).toBe(galleryPrint.metadata);
  });
});

describe("reconcileInterruptedGenerationJobs", () => {
  it("reconciles a structured interruption even when the transport copy is localized", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") return Promise.resolve({ entries: [] });
      if (path === "/api/gallery") return Promise.resolve([galleryPrint]);
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob({ error: "La conexión de red se perdió.", interrupted: true });

    await reconcileInterruptedGenerationJobs([job], options());

    expect(job.status).toBe("complete");
  });

  it("settles a job that finished server-side as a rendered completion", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") return Promise.resolve({ entries: [] });
      if (path === "/api/gallery") return Promise.resolve([galleryPrint]);
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob();
    const opts = options();
    await reconcileInterruptedGenerationJobs([job], opts);

    expect(job.status).toBe("complete");
    expect(job.error).toBeNull();
    expect(job.result?.filename).toBe("resumed print.png");
    expect(job.result?.seed_used).toBe(77);
    expect(opts.refreshResultUrl).toHaveBeenCalledWith(7);
  });

  it("removes a zombie queued row and explains the outcome in human copy", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        return Promise.resolve({
          entries: [{ id: "job-9", model: "ltx2:q8", state: "queued", position: 0 }],
        });
      }
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob();
    await reconcileInterruptedGenerationJobs([job], options());

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/job-9", { method: "DELETE" });
    expect(job.status).toBe("error");
    expect(job.error).toBe(
      "The connection dropped while this print waited in Studio’s queue. Develop again to requeue it.",
    );
  });

  it("re-attaches to a running job by polling until the print lands", async () => {
    let queueCalls = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        queueCalls += 1;
        return Promise.resolve({
          entries:
            queueCalls <= 2
              ? [{ id: "job-9", model: "ltx2:q8", state: "running", position: 0 }]
              : [],
        });
      }
      if (path === "/api/gallery") return Promise.resolve([galleryPrint]);
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob();
    const stages: Array<string | null> = [];
    const opts = options({
      sleep: () => {
        stages.push(job.stage);
        return Promise.resolve();
      },
    });
    await reconcileInterruptedGenerationJobs([job], opts);

    expect(queueCalls).toBe(3);
    expect(stages).toEqual(["Developing on Studio", "Developing on Studio"]);
    expect(job.status).toBe("complete");
    expect(job.result?.filename).toBe("resumed print.png");
  });

  it("joins an id-less job to its queue row by pinned seed before deleting it", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        return Promise.resolve({
          entries: [
            {
              id: "someone-elses",
              model: "ltx2:q8",
              state: "queued",
              position: 0,
              metadata: { ...galleryPrint.metadata, seed: 41 },
            },
            {
              id: "mine",
              model: "ltx2:q8",
              state: "queued",
              position: 1,
              started_at_unix_ms: job.submittedAtUnixMs + 100,
              metadata: galleryPrint.metadata,
            },
          ],
        });
      }
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob({ id: "" });
    await reconcileInterruptedGenerationJobs([job], options());

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/mine", { method: "DELETE" });
    expect(job.status).toBe("error");
  });

  it("does not delete a compatible pre-ID duplicate submitted after the interrupted job", async () => {
    const submittedAtUnixMs = 1_700_000_000_000;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        return Promise.resolve({
          entries: [
            {
              id: "mine",
              model: "ltx2:q8",
              state: "queued",
              started_at_unix_ms: submittedAtUnixMs + 100,
              metadata: galleryPrint.metadata,
            },
            {
              id: "later-duplicate",
              model: "ltx2:q8",
              state: "queued",
              started_at_unix_ms: submittedAtUnixMs + 10_000,
              metadata: galleryPrint.metadata,
            },
          ],
        });
      }
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob({ id: "", submittedAtUnixMs });

    await reconcileInterruptedGenerationJobs([job], options());

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/mine", { method: "DELETE" });
    expect(apiFetchTo).not.toHaveBeenCalledWith(target, "/api/queue/later-duplicate", {
      method: "DELETE",
    });
  });

  it("humanizes a reconciliation query that itself cannot reach the host", async () => {
    apiJsonTo.mockRejectedValue(new TypeError("Load failed"));
    const job = makeJob();
    await reconcileInterruptedGenerationJobs([job], options());

    expect(job.status).toBe("error");
    expect(job.error).toBe("Couldn’t reach Studio. Check the connection and try again.");
    // Bounded retry: the first query plus three re-attempts, never unbounded.
    expect(apiJsonTo).toHaveBeenCalledTimes(4);
  });

  it("retries a transport-failed reconciliation query and still lands the print", async () => {
    // Resume fires while iOS Wi-Fi is still re-associating: the first two
    // queries die exactly like the stream did, then the network returns.
    apiJsonTo
      .mockRejectedValueOnce(new TypeError("Load failed"))
      .mockRejectedValueOnce(new TypeError("Load failed"))
      .mockResolvedValueOnce({ entries: [] })
      .mockResolvedValueOnce([galleryPrint]);
    const job = makeJob();
    const opts = options();
    await reconcileInterruptedGenerationJobs([job], opts);

    expect(job.status).toBe("complete");
    expect(job.error).toBeNull();
    expect(opts.refreshResultUrl).toHaveBeenCalledWith(7);
  });

  it("does not let a transport retry overwrite an externally settled job", async () => {
    const job = makeJob();
    apiJsonTo.mockImplementation(() => {
      // The user cancels (external settle) while the retry backoff waits.
      job.status = "error";
      job.error = "Cancelled";
      return Promise.reject(new TypeError("Load failed"));
    });
    await reconcileInterruptedGenerationJobs([job], options());

    expect(job.error).toBe("Cancelled");
    expect(apiJsonTo).toHaveBeenCalledTimes(1);
  });

  it("reports a vanished job without transport jargon", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") return Promise.resolve({ entries: [] });
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob();
    await reconcileInterruptedGenerationJobs([job], options());

    expect(job.status).toBe("error");
    expect(job.error).toBe(
      "The connection to Studio was interrupted and this print didn’t finish.",
    );
  });

  it("never overwrites a job settled externally while reconciling", async () => {
    const job = makeJob();
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        // A user cancel wins while the queue query is in flight.
        job.status = "error";
        job.error = "Cancelled";
        return Promise.resolve({ entries: [] });
      }
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    await reconcileInterruptedGenerationJobs([job], options());

    expect(job.error).toBe("Cancelled");
    expect(apiJsonTo).toHaveBeenCalledTimes(1);
  });

  it("leaves server-authored failures and completions untouched", async () => {
    const failed = makeJob({ error: "host ran out of memory" });
    const cancelled = makeJob({ error: "Cancelled" });
    await reconcileInterruptedGenerationJobs([failed, cancelled], options());

    expect(failed.error).toBe("host ran out of memory");
    expect(cancelled.error).toBe("Cancelled");
    expect(apiJsonTo).not.toHaveBeenCalled();
  });

  it("resumes a durable chain job through its chain record", async () => {
    let chainCalls = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/chain-jobs/chain-4") {
        chainCalls += 1;
        return Promise.resolve(
          chainCalls === 1
            ? { id: "chain-4", state: "running", finalizes: [] }
            : {
                id: "chain-4",
                state: "completed",
                finalizes: [{ output: "stitched clip.mp4", at_unix_ms: 1, stage_seeds: [] }],
              },
        );
      }
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob({ id: "chain-4" });
    const opts = options({ chain: true });
    await reconcileInterruptedGenerationJobs([job], opts);

    expect(chainCalls).toBe(2);
    expect(job.status).toBe("complete");
    expect(job.result?.filename).toBe("stitched clip.mp4");
    expect(job.result?.format).toBe("mp4");
    expect(opts.refreshResultUrl).toHaveBeenCalledWith(7);
  });

  it("finds only the id-less one-shot shim created with the interrupted submission", async () => {
    const submittedAtUnixMs = 1_700_000_000_000;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/chain-jobs?include_ephemeral=true") {
        return Promise.resolve({
          jobs: [
            {
              id: "authored-sequence",
              state: "running",
              model: "ltx2:q8",
              created_at_unix_ms: submittedAtUnixMs + 50,
              ephemeral: false,
            },
            {
              id: "chain-later",
              state: "running",
              model: "ltx2:q8",
              created_at_unix_ms: submittedAtUnixMs + 200,
              ephemeral: true,
            },
            {
              id: "chain-mine",
              state: "running",
              model: "ltx2:q8",
              created_at_unix_ms: submittedAtUnixMs + 100,
              ephemeral: true,
            },
          ],
        });
      }
      if (path === "/api/chain-jobs/chain-mine") {
        return Promise.resolve({
          id: "chain-mine",
          state: "completed",
          finalizes: [{ output: "recovered.mp4", at_unix_ms: 1, stage_seeds: [] }],
        });
      }
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob({ id: "", submittedAtUnixMs });

    await reconcileInterruptedGenerationJobs([job], options({ chain: true }));

    expect(job.id).toBe("chain-mine");
    expect(job.status).toBe("complete");
    expect(job.result?.filename).toBe("recovered.mp4");
    expect(apiJsonTo).toHaveBeenCalledWith(target, "/api/chain-jobs?include_ephemeral=true");
  });

  it("surfaces a chain failure with the host's own error copy", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/chain-jobs/chain-4") {
        return Promise.resolve({
          id: "chain-4",
          state: "failed",
          error: "stage 2 ran out of VRAM",
          finalizes: [],
        });
      }
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob({ id: "chain-4" });
    await reconcileInterruptedGenerationJobs([job], options({ chain: true }));

    expect(job.status).toBe("error");
    expect(job.error).toBe("stage 2 ran out of VRAM");
  });

  it("falls back to the gallery when the chain record is gone", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/chain-jobs/chain-4") {
        return Promise.reject(new ApiError("not found", 404));
      }
      if (path === "/api/gallery") return Promise.resolve([galleryPrint]);
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob({ id: "chain-4" });
    await reconcileInterruptedGenerationJobs([job], options({ chain: true }));

    expect(job.status).toBe("complete");
    expect(job.result?.filename).toBe("resumed print.png");
  });
});
