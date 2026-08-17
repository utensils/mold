import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GalleryImage, GenerateRequest } from "./api/types";
import { newJob, type Job } from "./generationJob";

const { apiFetchTo, apiJsonTo } = vi.hoisted(() => ({
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(),
}));

vi.mock("./api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api/client")>()),
  apiFetchTo,
  apiJsonTo,
}));

vi.mock("./sourceRestore", async (importOriginal) => {
  const original = await importOriginal<typeof import("./sourceRestore")>();
  return {
    ...original,
    sha256HexOfBase64: vi.fn(original.sha256HexOfBase64),
  };
});

import { ApiError } from "./api/client";
import { sha256HexOfBase64 } from "./sourceRestore";
import {
  galleryCompletion,
  isInterruptedGenerationError,
  matchGalleryPrint,
  reconcileInterruptedGenerationJobs,
} from "./generationRecovery";

const target = { baseUrl: "http://studio.tailnet.ts.net:7680", apiKey: "secret" };
const ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
const DEF_SHA256 = "cb8379ac2098aa165029e3938a51da0bcecfc008fd6795f401178647f96c5b34";
const GHI_SHA256 = "50ae61e841fac4e8f9e40baf2ad36ec868922ea48368c18f9535e47db56dd7fb";
const sha256HexOfBase64Mock = vi.mocked(sha256HexOfBase64);

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

function makeH3Job(request: GenerateRequest, overrides: Partial<Job> = {}): Job {
  const job = newJob(request);
  return Object.assign(job, {
    clientId: 8,
    batchId: 2,
    id: "h3-job",
    hostId: "studio-id",
    hostLabel: "Studio",
    remote: true,
    metadataOnlyCompletion: true,
    streamStarted: true,
    status: "error" as const,
    error: "Load failed",
    ...overrides,
  });
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
  sha256HexOfBase64Mock.mockClear();
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
  it("joins on the explicit seed and model", async () => {
    const job = makeJob();
    expect(await matchGalleryPrint(job, [galleryPrint])).toBe(galleryPrint);
    expect(await matchGalleryPrint(makeJob({ visualSeed: "78" }), [galleryPrint])).toBeNull();
    expect(await matchGalleryPrint(makeJob({ model: "flux:q8" }), [galleryPrint])).toBeNull();
  });

  it("uses the recorded prompt as the prepared-sibling tiebreaker", async () => {
    expect(
      await matchGalleryPrint(makeJob({ prompt: "a different sibling" }), [galleryPrint]),
    ).toBeNull();
  });

  it("rejects a stale same-seed print whose dimensions or steps differ", async () => {
    // A fixed-seed re-run at a new resolution must not resurrect yesterday's
    // print: the join requires dims and steps to agree when both sides know
    // them.
    expect(
      await matchGalleryPrint(makeJob({ width: 1024, height: 1024 }), [galleryPrint]),
    ).toBeNull();
    expect(await matchGalleryPrint(makeJob({ total: 8 }), [galleryPrint])).toBeNull();
    // Absent metadata on either side stays permissive — old hosts omit steps.
    const { steps: _s, width: _w, height: _h, ...bareMeta } = galleryPrint.metadata;
    const bare = { ...galleryPrint, metadata: bareMeta } as GalleryImage;
    expect(await matchGalleryPrint(makeJob({ width: 1024, height: 1024 }), [bare])).toBe(bare);
  });

  it("requires an exact numeric seed", async () => {
    expect(
      await matchGalleryPrint(makeJob({ visualSeed: "ltx2:q8·prompt" }), [galleryPrint]),
    ).toBeNull();
  });

  it("binds H3 endpoint gallery recovery to exact first- and last-frame hashes", async () => {
    const request: GenerateRequest = {
      prompt: "a synchronized storm crossing",
      model: "minimax-h3-fl2va:official-bf16",
      width: 768,
      height: 512,
      steps: 50,
      frames: 125,
      fps: 24,
      seed: 77,
      source_image: "YWJj",
      source_image_name: "opening.png",
      keyframes: [{ frame: 124, image: "ZGVm", name: "closing.png" }],
    };
    const job = makeH3Job(request);
    const metadata = {
      prompt: request.prompt,
      model: request.model,
      seed: 77,
      steps: 50,
      guidance: 0,
      width: 768,
      height: 512,
      frames: 125,
      fps: 24,
      source_image_name: "opening.png",
      source_image_sha256: ABC_SHA256,
      keyframes: [{ frame: 124, name: "closing.png", sha256: DEF_SHA256 }],
    };
    const wrongFirst = {
      filename: "wrong-first.mp4",
      timestamp: 1,
      format: "mp4" as const,
      metadata: { ...metadata, source_image_sha256: GHI_SHA256 },
    };
    const wrongLast = {
      filename: "wrong-last.mp4",
      timestamp: 2,
      format: "mp4" as const,
      metadata: {
        ...metadata,
        keyframes: [{ frame: 124, name: "closing.png", sha256: GHI_SHA256 }],
      },
    };
    const exact = {
      filename: "exact.mp4",
      timestamp: 3,
      format: "mp4" as const,
      metadata,
    };

    expect(await matchGalleryPrint(job, [wrongFirst, wrongLast, exact])).toBe(exact);
    expect(await matchGalleryPrint(job, [wrongFirst, wrongLast])).toBeNull();
  });

  it("fails H3 gallery recovery closed on missing or mismatched shape and timing", async () => {
    const request: GenerateRequest = {
      prompt: "a synchronized storm crossing",
      model: "minimax-h3-fl2va:official-bf16",
      width: 768,
      height: 512,
      steps: 50,
      frames: 125,
      fps: 24,
      seed: 77,
      source_image: "YWJj",
    };
    const job = makeH3Job(request);
    const metadata = {
      prompt: request.prompt,
      model: request.model,
      seed: 77,
      steps: 50,
      guidance: 0,
      width: 768,
      height: 512,
      frames: 125,
      fps: 24,
      source_image_sha256: ABC_SHA256,
    };
    const print = {
      filename: "exact.mp4",
      timestamp: 1,
      format: "mp4" as const,
      metadata,
    };

    expect(await matchGalleryPrint(job, [print])).toBe(print);
    for (const field of ["width", "height", "steps", "frames", "fps"] as const) {
      const missing = { ...metadata } as Partial<typeof metadata>;
      delete missing[field];
      expect(
        await matchGalleryPrint(job, [{ ...print, metadata: missing } as GalleryImage]),
      ).toBeNull();
      expect(
        await matchGalleryPrint(job, [
          { ...print, metadata: { ...metadata, [field]: metadata[field] + 1 } },
        ]),
      ).toBeNull();
    }
    const { source_image_sha256: _sha, ...missingConditioning } = metadata;
    expect(
      await matchGalleryPrint(job, [{ ...print, metadata: missingConditioning } as GalleryImage]),
    ).toBeNull();

    const { frames: _frames, ...withoutFrames } = request;
    const { fps: _fps, ...withoutFps } = request;
    const missingSubmittedFrames = makeH3Job(withoutFrames);
    const missingSubmittedFps = makeH3Job(withoutFps);
    expect(await matchGalleryPrint(missingSubmittedFrames, [print])).toBeNull();
    expect(await matchGalleryPrint(missingSubmittedFps, [print])).toBeNull();
  });

  it("binds H3 Ref2VA gallery recovery to ordered reference hashes", async () => {
    const request: GenerateRequest = {
      prompt: "resynthesize the performance",
      model: "minimax-h3-ref2va:official-bf16",
      width: 768,
      height: 512,
      steps: 50,
      frames: 125,
      fps: 24,
      seed: 77,
      references: [
        {
          kind: "image",
          media: { authority: "inline", data: "YWJj" },
          provenance: { name: "subject.png", sha256: ABC_SHA256 },
          mime_type: "image/png",
          width: 512,
          height: 512,
        },
        {
          kind: "audio",
          media: { authority: "inline", data: "ZGVm" },
          provenance: { name: "voice.wav", sha256: DEF_SHA256 },
          mime_type: "audio/wav",
          duration_ms: 2_000,
          sample_rate: 32_000,
          channels: 2,
          sample_count: 64_000,
        },
      ],
    };
    const job = makeH3Job(request);
    const metadata = {
      prompt: request.prompt,
      model: request.model,
      seed: 77,
      steps: 50,
      guidance: 0,
      width: 768,
      height: 512,
      frames: 125,
      fps: 24,
      references: [
        {
          index: 1,
          kind: "image" as const,
          name: "subject.png",
          sha256: ABC_SHA256,
          mime_type: "image/png",
        },
        {
          index: 2,
          kind: "audio" as const,
          name: "voice.wav",
          sha256: DEF_SHA256,
          mime_type: "audio/wav",
        },
      ],
    };
    const reordered = {
      filename: "reordered.mp4",
      timestamp: 1,
      format: "mp4" as const,
      metadata: {
        ...metadata,
        references: [
          { ...metadata.references[1]!, index: 1 },
          { ...metadata.references[0]!, index: 2 },
        ],
      },
    };
    const exact = {
      filename: "exact.mp4",
      timestamp: 2,
      format: "mp4" as const,
      metadata,
    };

    expect(await matchGalleryPrint(job, [reordered, exact])).toBe(exact);
    expect(await matchGalleryPrint(job, [reordered])).toBeNull();
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

  it("keeps a queued row waiting when THAT JOB is durable on the host", async () => {
    let queueCalls = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        queueCalls += 1;
        return Promise.resolve({
          entries:
            queueCalls <= 2
              ? [{ id: "job-9", model: "ltx2:q8", state: "queued", position: 0, durable: true }]
              : [],
        });
      }
      if (path === "/api/gallery") return Promise.resolve([galleryPrint]);
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob();
    const stages: Array<string | null> = [];
    await reconcileInterruptedGenerationJobs(
      [job],
      options({
        sleep: () => {
          stages.push(job.stage);
          return Promise.resolve();
        },
      }),
    );

    // The host runs this job with no client attached — deleting it would
    // destroy exactly the work the host kept. Per-job truth, and no extra
    // request: `/api/queue` already carries it.
    expect(apiFetchTo).not.toHaveBeenCalled();
    expect(apiJsonTo).not.toHaveBeenCalledWith(target, "/api/capabilities");
    expect(stages).toEqual(["Waiting in Studio’s queue", "Waiting in Studio’s queue"]);
    expect(job.status).toBe("complete");
    expect(job.result?.filename).toBe("resumed print.png");
  });

  it("clears a queued row the durable host did not journal", async () => {
    // `durable: false` on a durable host — no gallery target, reference-upload
    // media, or an oversized request. Host capability alone would over-promise
    // and hang this row forever.
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        return Promise.resolve({
          entries: [
            { id: "job-9", model: "ltx2:q8", state: "queued", position: 0, durable: false },
          ],
        });
      }
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob();
    await reconcileInterruptedGenerationJobs([job], options());

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/job-9", { method: "DELETE" });
    expect(job.status).toBe("error");
  });

  it("waits out the handoff gap when a retained job is briefly absent", async () => {
    // Between the retained worker exiting and restart replay running, the
    // durable row is invisible to /api/queue. A successful EMPTY listing in
    // that window is a moment of server bookkeeping, not evidence the work is
    // gone — and `retainedByHost` is proof the host said it kept it.
    let queuePolls = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        queuePolls += 1;
        // Absent for two polls (the handoff), then replayed as queued, then
        // gone for real once it has finished.
        if (queuePolls <= 2) return Promise.resolve({ entries: [] });
        if (queuePolls === 3) {
          return Promise.resolve({
            entries: [
              { id: "job-9", model: "ltx2:q8", state: "queued", position: 0, durable: true },
            ],
          });
        }
        return Promise.resolve({ entries: [] });
      }
      if (path === "/api/gallery") {
        // The print only exists after the replay actually ran.
        return Promise.resolve(queuePolls >= 4 ? [galleryPrint] : []);
      }
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob({ retainedByHost: true });
    await reconcileInterruptedGenerationJobs([job], options());

    expect(job.status).toBe("complete");
    expect(job.result?.filename).toBe("resumed print.png");
  });

  it("still fails a NON-retained job that is absent with no print", async () => {
    // Without the host's promise, an empty queue and an empty gallery is the
    // only evidence there is, and it says the work is gone.
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

  it("gives up on a retained job that never comes back, without claiming it failed", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") return Promise.resolve({ entries: [] });
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob({ retainedByHost: true });
    await reconcileInterruptedGenerationJobs([job], options());

    // Bounded, so reconciliation always ends — but the copy still points at
    // the host, and the row stays reconcilable rather than declared failed.
    expect(job.status).toBe("error");
    expect(job.interrupted).toBe(true);
    expect(job.error).toContain("check the Library");
  });

  it("settles a held row with the host's reason instead of waiting forever", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        return Promise.resolve({
          entries: [
            {
              id: "job-9",
              model: "ltx2:q8",
              state: "held",
              position: 0,
              durable: true,
              held_reason: "dispatch attempts exhausted",
            },
          ],
        });
      }
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeJob();
    await reconcileInterruptedGenerationJobs([job], options());

    // A held row is listed but never auto-run, so waiting on it never ends.
    expect(apiFetchTo).not.toHaveBeenCalled();
    expect(job.status).toBe("error");
    expect(job.error).toBe(
      "Studio is holding this print and will not run it automatically (dispatch attempts exhausted). Develop again to requeue it.",
    );
  });

  it("reaches no verdict when the host never answers, and says so", async () => {
    // Three dead queries in a row is not evidence the print failed — the
    // client simply never learned anything. The row stays flagged as an
    // interruption so a later resume can try again and callers that use the
    // flag to suppress failure announcements keep doing so.
    apiJsonTo.mockRejectedValue(new TypeError("Load failed"));
    const job = makeJob();
    await reconcileInterruptedGenerationJobs([job], options());

    expect(job.status).toBe("error");
    expect(job.interrupted).toBe(true);
    expect(apiFetchTo).not.toHaveBeenCalled();
  });

  it("reaches a verdict when the host answers that the work is gone", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) =>
      Promise.resolve(path === "/api/queue" ? { entries: [] } : []),
    );
    const job = makeJob();
    await reconcileInterruptedGenerationJobs([job], options());

    // The host answered: no such job, no such print. That IS a failure, and
    // the flag clears so it is announced like one.
    expect(job.status).toBe("error");
    expect(job.interrupted).toBe(false);
    expect(job.error).toBe(
      "The connection to Studio was interrupted and this print didn’t finish.",
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

  it("joins an id-less H3 job only to the queue row with matching conditioning hashes", async () => {
    const submittedAtUnixMs = 1_700_000_000_000;
    const request: GenerateRequest = {
      prompt: "a synchronized storm crossing",
      model: "minimax-h3-fl2va:official-bf16",
      width: 768,
      height: 512,
      steps: 50,
      frames: 125,
      fps: 24,
      seed: 77,
      source_image: "YWJj",
      keyframes: [{ frame: 124, image: "ZGVm", name: "closing.png" }],
    };
    const metadata = {
      prompt: request.prompt,
      model: request.model,
      seed: 77,
      steps: 50,
      guidance: 0,
      width: 768,
      height: 512,
      frames: 125,
      fps: 24,
      source_image_sha256: ABC_SHA256,
      keyframes: [{ frame: 124, name: "closing.png", sha256: DEF_SHA256 }],
    };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        return Promise.resolve({
          entries: [
            {
              id: "same-settings-missing-shape",
              model: request.model,
              state: "queued",
              started_at_unix_ms: submittedAtUnixMs + 10,
              metadata: { ...metadata, width: undefined },
            },
            {
              id: "same-settings-missing-frames",
              model: request.model,
              state: "queued",
              started_at_unix_ms: submittedAtUnixMs + 20,
              metadata: { ...metadata, frames: undefined },
            },
            {
              id: "same-settings-wrong-fps",
              model: request.model,
              state: "queued",
              started_at_unix_ms: submittedAtUnixMs + 30,
              metadata: { ...metadata, fps: 30 },
            },
            {
              id: "same-settings-wrong-conditioning",
              model: request.model,
              state: "queued",
              started_at_unix_ms: submittedAtUnixMs + 50,
              metadata: { ...metadata, source_image_sha256: GHI_SHA256 },
            },
            {
              id: "h3-mine",
              model: request.model,
              state: "queued",
              started_at_unix_ms: submittedAtUnixMs + 100,
              metadata,
            },
          ],
        });
      }
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeH3Job(request, { id: "", submittedAtUnixMs });

    await reconcileInterruptedGenerationJobs([job], options());

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/h3-mine", {
      method: "DELETE",
    });
    expect(apiFetchTo).not.toHaveBeenCalledWith(
      target,
      "/api/queue/same-settings-wrong-conditioning",
      { method: "DELETE" },
    );
    for (const id of [
      "same-settings-missing-shape",
      "same-settings-missing-frames",
      "same-settings-wrong-fps",
    ]) {
      expect(apiFetchTo).not.toHaveBeenCalledWith(target, `/api/queue/${id}`, {
        method: "DELETE",
      });
    }
  });

  it("hashes H3 conditioning once across repeated queue polls and gallery recovery", async () => {
    const submittedAtUnixMs = 1_700_000_000_000;
    const request: GenerateRequest = {
      prompt: "a synchronized storm crossing",
      model: "minimax-h3-fl2va:official-bf16",
      width: 768,
      height: 512,
      steps: 50,
      frames: 125,
      fps: 24,
      seed: 77,
      source_image: "YWJj",
      keyframes: [{ frame: 124, image: "ZGVm", name: "closing.png" }],
    };
    const metadata = {
      prompt: request.prompt,
      model: request.model,
      seed: 77,
      steps: 50,
      guidance: 0,
      width: 768,
      height: 512,
      frames: 125,
      fps: 24,
      source_image_sha256: ABC_SHA256,
      keyframes: [{ frame: 124, name: "closing.png", sha256: DEF_SHA256 }],
    };
    const print = {
      filename: "exact.mp4",
      timestamp: 1,
      format: "mp4" as const,
      metadata,
    };
    let queueCalls = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/queue") {
        queueCalls += 1;
        return Promise.resolve({
          entries:
            queueCalls <= 2
              ? [
                  {
                    id: "h3-running",
                    model: request.model,
                    state: "running",
                    started_at_unix_ms: submittedAtUnixMs + 100,
                    metadata,
                  },
                ]
              : [],
        });
      }
      if (path === "/api/gallery") return Promise.resolve([print]);
      return Promise.reject(new Error(`Unexpected path ${path}`));
    });
    const job = makeH3Job(request, { id: "", submittedAtUnixMs });

    await reconcileInterruptedGenerationJobs([job], options());

    expect(queueCalls).toBe(3);
    expect(sha256HexOfBase64Mock.mock.calls).toEqual([["YWJj"], ["ZGVm"]]);
    expect(job.status).toBe("complete");
    expect(job.result?.filename).toBe("exact.mp4");
  });

  it("lets cancellation win while the initial H3 digest is in flight", async () => {
    let releaseDigest!: (value: string) => void;
    sha256HexOfBase64Mock.mockImplementationOnce(
      () =>
        new Promise<string>((resolve) => {
          releaseDigest = resolve;
        }),
    );
    const request: GenerateRequest = {
      prompt: "a synchronized storm crossing",
      model: "minimax-h3-fl2va:official-bf16",
      width: 768,
      height: 512,
      steps: 50,
      frames: 125,
      fps: 24,
      seed: 77,
      source_image: "YWJj",
    };
    const job = makeH3Job(request);
    const opts = options();

    const recovery = reconcileInterruptedGenerationJobs([job], opts);
    await vi.waitFor(() => expect(sha256HexOfBase64Mock).toHaveBeenCalledOnce());
    job.status = "error";
    job.error = "Cancelled";
    releaseDigest(ABC_SHA256);
    await recovery;

    expect(job.status).toBe("error");
    expect(job.error).toBe("Cancelled");
    expect(apiJsonTo).not.toHaveBeenCalled();
    expect(opts.refreshResultUrl).not.toHaveBeenCalled();
  });

  it("abandons recovery when the owner unmounts during the initial H3 digest", async () => {
    let releaseDigest!: (value: string) => void;
    let active = true;
    sha256HexOfBase64Mock.mockImplementationOnce(
      () =>
        new Promise<string>((resolve) => {
          releaseDigest = resolve;
        }),
    );
    const request: GenerateRequest = {
      prompt: "a synchronized storm crossing",
      model: "minimax-h3-fl2va:official-bf16",
      width: 768,
      height: 512,
      steps: 50,
      frames: 125,
      fps: 24,
      seed: 77,
      source_image: "YWJj",
    };
    const job = makeH3Job(request);
    const opts = options({ isActive: () => active });

    const recovery = reconcileInterruptedGenerationJobs([job], opts);
    await vi.waitFor(() => expect(sha256HexOfBase64Mock).toHaveBeenCalledOnce());
    active = false;
    releaseDigest(ABC_SHA256);
    await recovery;

    expect(job.status).toBe("loading");
    expect(job.error).toBeNull();
    expect(apiJsonTo).not.toHaveBeenCalled();
    expect(opts.refreshResultUrl).not.toHaveBeenCalled();
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
