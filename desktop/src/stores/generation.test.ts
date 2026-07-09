import { describe, expect, it } from "vitest";
import { applyProgress, jobPhase, jobProgress, newJob } from "./generation";

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
