import { describe, expect, it } from "vitest";
import {
  applyProgress,
  jobPhase,
  jobProgress,
  jobStatusCode,
  newJob,
  planBatchRequests,
  resolveBaseSeed,
  type Job,
  type JobStatus,
} from "./generation";

const req = {
  prompt: "a lighthouse at dusk",
  model: "flux-schnell:q8",
  width: 1024,
  height: 1024,
  steps: 4,
};

describe("generation SSE reducer", () => {
  it("walks queued → loading → denoising with server-assigned id", () => {
    const job = newJob(req);
    expect(jobPhase(job)).toBe("latent");

    applyProgress(job, { type: "queued", position: 2, id: "abc" });
    expect(job.queuePosition).toBe(2);
    expect(job.id).toBe("abc");

    applyProgress(job, {
      type: "weight_load",
      bytes_loaded: 1,
      bytes_total: 10,
      component: "t5",
    });
    expect(job.status).toBe("loading");

    applyProgress(job, { type: "denoise_step", step: 2, total: 4, elapsed_ms: 100 });
    expect(job.status).toBe("denoising");
    expect(jobPhase(job)).toBe("developing");
    expect(jobProgress(job)).toBe(0.5);
    expect(job.queuePosition).toBeNull();
  });

  it("post-denoise stages become 'finishing', never regress to loading", () => {
    // After the last denoise step the engine drops the transformer, loads
    // the VAE, and decodes — steps read 4/4 but the print isn't done. Those
    // trailing stage events are the fixer bath.
    const job = newJob(req);
    applyProgress(job, { type: "denoise_step", step: 4, total: 4, elapsed_ms: 1 });
    applyProgress(job, { type: "stage_start", name: "VAE decode" });
    expect(job.status).toBe("finishing");
    expect(job.stage).toBe("VAE decode");
    expect(jobPhase(job)).toBe("developing");
    expect(jobProgress(job)).toBe(1);
    // weight_load during finishing must not flip it back to loading either.
    applyProgress(job, {
      type: "weight_load",
      bytes_loaded: 1,
      bytes_total: 2,
      component: "vae",
    });
    expect(job.status).toBe("finishing");
  });

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
  });

  it("preserves queue, loading, and progress detail", () => {
    expect(jobStatusCode({ ...withStatus("queued"), queuePosition: 2 })).toBe("QUEUED #2");
    expect(jobStatusCode(withStatus("loading"))).toBe("LOADING");
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

  it("clamps sub-1 and fractional batch sizes to at least one job", () => {
    expect(planBatchRequests(req, 0, 5)).toHaveLength(1);
    expect(planBatchRequests(req, 2.9, 5).map((p) => p.seed)).toEqual([5, 6]);
  });

  it("does not mutate the source request", () => {
    const snapshot = JSON.stringify(req);
    planBatchRequests(req, 3, 7);
    expect(JSON.stringify(req)).toBe(snapshot);
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

describe("live latent previews", () => {
  it("preview events advance to denoising and hold an object URL", () => {
    const job = newJob(req);
    applyProgress(job, { type: "preview", image: btoa("fake-png"), step: 1, total: 4 });
    expect(job.status).toBe("denoising");
    expect(job.previewUrl).not.toBeNull();
  });

  it("each preview revokes the previous frame's URL", async () => {
    const { vi } = await import("vitest");
    const revoke = vi.spyOn(URL, "revokeObjectURL");
    const job = newJob(req);
    applyProgress(job, { type: "preview", image: btoa("a"), step: 1, total: 4 });
    const first = job.previewUrl;
    applyProgress(job, { type: "preview", image: btoa("b"), step: 2, total: 4 });
    expect(revoke).toHaveBeenCalledWith(first);
    expect(job.previewUrl).not.toBe(first);
    revoke.mockRestore();
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
