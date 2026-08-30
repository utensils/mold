import { describe, expect, it } from "vitest";
import { newJob } from "./generationJob";
import { applyDurablePresentation } from "./durableGenerationPresentation";

const request = {
  model: "flux2-klein-9b:bf16",
  prompt: "a frog",
  width: 1024,
  height: 1024,
  steps: 4,
};

describe("desktop durable generation presentation", () => {
  it("keeps the host's running sub-stage across lifecycle reconciliation", () => {
    const job = newJob(request);
    job.status = "loading";
    job.stage = "Loading model";

    applyDurablePresentation(job, { kind: "running", label: "Developing" });

    expect(job.stage).toBe("Loading model");
  });

  it("sets the generic running stage on the first worker lease", () => {
    const job = newJob(request);

    applyDurablePresentation(job, { kind: "running", label: "Developing" });

    expect(job.status).toBe("loading");
    expect(job.stage).toBe("Developing");
  });
});
