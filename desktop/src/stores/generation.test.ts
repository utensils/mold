import { describe, expect, it } from "vitest";
import {
  applyProgress,
  jobPhase,
  jobProgress,
  newJob,
  planBatchRequests,
  resolveBaseSeed,
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

  it("a late stage_start does not regress denoising status", () => {
    const job = newJob(req);
    applyProgress(job, { type: "denoise_step", step: 1, total: 4, elapsed_ms: 1 });
    applyProgress(job, { type: "stage_start", name: "vae" });
    expect(job.status).toBe("denoising");
    expect(job.stage).toBe("vae");
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
