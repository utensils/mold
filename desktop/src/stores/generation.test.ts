import { describe, expect, it } from "vitest";
import {
  jobProgress,
  jobStatusCode,
  newJob,
  planBatchRequests,
  resolveBaseSeed,
  type Job,
  type JobStatus,
} from "./generation";
// `applyCompletionWarnings` is a lib helper: the store no longer re-exports it
// now that no completion arrives on a stream.
import { applyCompletionWarnings } from "../lib/generationJob";

const req = {
  prompt: "a lighthouse at dusk",
  model: "flux-schnell:q8",
  width: 1024,
  height: 1024,
  steps: 4,
};

describe("generation SSE reducer", () => {
  it("uses the requested seed for the grain, or a stable stand-in", () => {
    expect(newJob({ ...req, seed: 42 }).visualSeed).toBe("42");
    const a = newJob(req).visualSeed;
    const b = newJob(req).visualSeed;
    expect(a).toBe(b);
  });

  it("total steps default to the requested steps before the first event", () => {
    const job = newJob(req);
    expect(job.total).toBe(4);
    expect(jobProgress(job)).toBe(0);
  });
});

describe("job status labels", () => {
  function withStatus(status: JobStatus, error: string | null = null): Job {
    return { ...newJob(req), status, error };
  }

  it("uses plain-language labels for terminal and finalizing states", () => {
    expect(jobStatusCode(withStatus("finishing"))).toBe("FINALIZING");
    expect(jobStatusCode(withStatus("complete"))).toBe("DONE");
    expect(jobStatusCode(withStatus("error", "out of memory"))).toBe("FAILED");
    expect(jobStatusCode(withStatus("error", "Cancelled"))).toBe("CANCELLED");
    expect(jobStatusCode(withStatus("error", "generation cancelled"))).toBe("CANCELLED");
  });

  it("preserves queue, loading, and progress detail", () => {
    expect(jobStatusCode({ ...withStatus("queued"), queuePosition: 2 })).toBe("QUEUED #2");
    expect(jobStatusCode(withStatus("loading"))).toBe("PREPARING");
    expect(jobStatusCode({ ...withStatus("loading"), stage: "Loading weights" })).toBe(
      "LOADING WEIGHTS",
    );
    expect(jobStatusCode({ ...withStatus("denoising"), step: 3, total: 8 })).toBe("3/8");
  });

  it("keeps malformed or future runtime statuses readable", () => {
    const unexpected = { ...withStatus("queued"), status: "future" } as unknown as Job;
    expect(jobStatusCode(unexpected)).toBe("UNKNOWN");
  });
});

describe("batch sequencing", () => {
  it("resolves an explicit finite seed and only draws random for none", () => {
    expect(resolveBaseSeed(42, () => 999)).toBe(42);
    expect(resolveBaseSeed(0, () => 999)).toBe(0);
    expect(resolveBaseSeed(undefined, () => 999)).toBe(999);
    expect(resolveBaseSeed(Number.NaN, () => 999)).toBe(999);
  });

  it("expands to base+i sibling seeds, each forced to batch_size 1", () => {
    const plans = planBatchRequests(req, 4, 100);
    expect(plans.map((p) => p.seed)).toEqual([100, 101, 102, 103]);
    expect(plans.every((p) => p.batch_size === 1)).toBe(true);
    // The rest of the request is carried through unchanged.
    expect(plans[0]!.prompt).toBe(req.prompt);
    expect(plans[0]!.model).toBe(req.model);
  });

  it("gives every sibling the same File under choice, prepared ones included", () => {
    const filed = { ...req, title: "Smurf Village", tags: ["smurf-village", "blue"] };
    const plain = planBatchRequests(filed, 3, 100);
    expect(plain.map((p) => p.title)).toEqual(["Smurf Village", "Smurf Village", "Smurf Village"]);
    expect(plain.map((p) => p.tags)).toEqual([
      ["smurf-village", "blue"],
      ["smurf-village", "blue"],
      ["smurf-village", "blue"],
    ]);
    const prepared = planBatchRequests({ ...filed, collection: { name: "River studies" } }, 2, 1, {
      prompts: ["a", "b"],
      batchId: "batch-abc",
    });
    expect(prepared.map((p) => p.collection)).toEqual([
      { name: "River studies" },
      { name: "River studies" },
    ]);
  });

  it("maps ordered prompts to seeds with one shared original prompt", () => {
    const prompts = ["storm-lit lighthouse", "lighthouse through sea mist", "aerial coast"];
    const plans = planBatchRequests(req, 3, 40, {
      prompts,
      originalPrompt: "a lighthouse at dusk",
      batchId: "batch-abc",
    });

    expect(plans).toEqual([
      {
        ...req,
        prompt: prompts[0],
        original_prompt: req.prompt,
        batch_id: "batch-abc",
        batch_index: 1,
        batch_count: 3,
        seed: 40,
        batch_size: 1,
      },
      {
        ...req,
        prompt: prompts[1],
        original_prompt: req.prompt,
        batch_id: "batch-abc",
        batch_index: 2,
        batch_count: 3,
        seed: 41,
        batch_size: 1,
      },
      {
        ...req,
        prompt: prompts[2],
        original_prompt: req.prompt,
        batch_id: "batch-abc",
        batch_index: 3,
        batch_count: 3,
        seed: 42,
        batch_size: 1,
      },
    ]);
  });

  it("preserves exact count, order, and provenance for a large prepared batch", () => {
    const prompts = Array.from({ length: 1_000 }, (_, index) => `variation ${index + 1}`);
    const plans = planBatchRequests(req, prompts.length, 500, {
      prompts,
      originalPrompt: req.prompt,
      batchId: "batch-large",
    });

    expect(plans).toHaveLength(1_000);
    expect(plans[0]).toMatchObject({
      prompt: "variation 1",
      original_prompt: req.prompt,
      batch_id: "batch-large",
      batch_index: 1,
      batch_count: 1_000,
      seed: 500,
      batch_size: 1,
    });
    expect(plans[999]).toMatchObject({
      prompt: "variation 1000",
      original_prompt: req.prompt,
      batch_id: "batch-large",
      batch_index: 1_000,
      batch_count: 1_000,
      seed: 1_499,
      batch_size: 1,
    });
  });

  it("rejects prompt lists that do not exactly match the normalized batch size", () => {
    expect(() =>
      planBatchRequests(req, 3, 7, {
        prompts: ["only one", "only two"],
        originalPrompt: req.prompt,
      }),
    ).toThrow("Per-item prompt count 2 does not match batch size 3");
    expect(() => planBatchRequests(req, 2.9, 7, { prompts: ["one", "two", "three"] })).toThrow(
      "Per-item prompt count 3 does not match batch size 2",
    );
  });

  it("clamps sub-1 and fractional batch sizes to at least one job", () => {
    expect(planBatchRequests(req, 0, 5)).toHaveLength(1);
    expect(planBatchRequests(req, 2.9, 5).map((p) => p.seed)).toEqual([5, 6]);
  });

  it("does not mutate the source request", () => {
    const snapshot = JSON.stringify(req);
    planBatchRequests(req, 3, 7);
    expect(JSON.stringify(req)).toBe(snapshot);
  });

  it("does not mutate per-item prompt inputs or share request objects", () => {
    const prompts = ["one", "two"];
    const promptSnapshot = [...prompts];
    const plans = planBatchRequests(req, 2, 7, {
      prompts,
      originalPrompt: "source",
    });

    plans[0]!.prompt = "changed after planning";
    expect(prompts).toEqual(promptSnapshot);
    expect(plans[1]!.prompt).toBe("two");
    expect(req.prompt).toBe("a lighthouse at dusk");
  });
});

describe("job reactivity wiring", () => {
  it("startJob returns the exact reference the UI reads via store.active", async () => {
    // Regression: startJob once returned the raw newJob() object while the
    // store held Vue's reactive proxy. The SSE closures mutated the raw
    // object, so no proxy trap ever fired and the canvas sat frozen at
    // "Queued 0/N" for the whole generation.
    const { createPinia, setActivePinia } = await import("pinia");
    const { useGenerationStore } = await import("./generation");
    setActivePinia(createPinia());
    const store = useGenerationStore();
    const job = store.startJob({ ...req });
    expect(job).toBe(store.active);

    const { isReactive } = await import("vue");
    expect(isReactive(job)).toBe(true);
  });
});

describe("railOrder", () => {
  const job = (clientId: number, status: JobStatus, queuePosition: number | null): Job => ({
    ...newJob(req),
    clientId,
    status,
    queuePosition,
  });

  it("shows developing first, then the server's queue order — not submission order", async () => {
    const { railOrder } = await import("./generation");
    // Concurrent batch submissions raced: job 1 landed at position #3,
    // job 3 at #1. The rail must show the engine's actual run order.
    const jobs = [
      job(1, "queued", 3),
      job(2, "queued", 2),
      job(3, "queued", 1),
      job(4, "denoising", null),
    ];
    expect(railOrder(jobs).map((j) => j.clientId)).toEqual([4, 3, 2, 1]);
  });

  it("positionless queued jobs sink below positioned ones, stable by clientId", async () => {
    const { railOrder } = await import("./generation");
    const jobs = [job(5, "queued", null), job(6, "queued", 1), job(7, "loading", null)];
    expect(railOrder(jobs).map((j) => j.clientId)).toEqual([7, 6, 5]);
  });
});

describe("completion advisories", () => {
  // The identity extractor decides which of several detected faces to
  // condition on while the job's dependencies are prepared — after `onOpen`
  // read the response headers, so the completion event is the only channel it
  // can arrive on. Reading headers alone dropped the notice entirely.
  it("appends advisories the render produced to the header ones", () => {
    const job = newJob(req);
    job.requestWarnings = ["the requested collection was dropped"];

    applyCompletionWarnings(job, {
      request_warnings: [
        "3 faces were detected in the identity image; conditioning on the largest one",
      ],
    });

    expect(job.requestWarnings).toEqual([
      "the requested collection was dropped",
      "3 faces were detected in the identity image; conditioning on the largest one",
    ]);
  });

  it("shows a repeated advisory once", () => {
    const job = newJob(req);
    job.requestWarnings = ["a lip dub was retimed"];
    applyCompletionWarnings(job, { request_warnings: ["a lip dub was retimed"] });
    expect(job.requestWarnings).toEqual(["a lip dub was retimed"]);
  });

  // The field is additive: an older server omits it, an ordinary render sends
  // nothing, and neither may disturb what the headers already said.
  it("leaves the header advisories alone when the render said nothing", () => {
    const job = newJob(req);
    job.requestWarnings = ["a lip dub was retimed"];
    applyCompletionWarnings(job, { seed_used: 7 });
    applyCompletionWarnings(job, { request_warnings: [] });
    applyCompletionWarnings(job, null);
    expect(job.requestWarnings).toEqual(["a lip dub was retimed"]);
  });
});
