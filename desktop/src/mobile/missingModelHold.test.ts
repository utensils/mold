import { describe, expect, it } from "vitest";
import { planHeldMissingModelPull } from "./missingModelHold";

const base = {
  jobId: "job-1",
  model: "z-image-turbo:q6",
  alreadyOffered: new Set<string>(),
};

describe("planHeldMissingModelPull", () => {
  it("offers the pull for a child the machine parked because the model is missing", () => {
    for (const heldReason of [
      "UNKNOWN_MODEL",
      "UNKNOWN_MODEL: z-image-turbo:q6 is not installed",
      "MODEL_NOT_FOUND while preparing",
    ]) {
      expect(planHeldMissingModelPull({ ...base, heldReason })).toEqual({
        model: "z-image-turbo:q6",
        jobId: "job-1",
      });
    }
  });

  it("offers nothing for a hold that is not about the model", () => {
    for (const heldReason of [
      null,
      "",
      "insufficient VRAM on this device",
      "QUEUE_PAUSED",
      "the operator disabled every device",
    ]) {
      expect(planHeldMissingModelPull({ ...base, heldReason })).toBeNull();
    }
  });

  /** A machine re-reports the same hold on every reconciliation wave. */
  it("offers a parked print exactly once", () => {
    const alreadyOffered = new Set<string>();
    const first = planHeldMissingModelPull({
      ...base,
      heldReason: "UNKNOWN_MODEL",
      alreadyOffered,
    });
    expect(first).not.toBeNull();
    alreadyOffered.add(first!.jobId);
    expect(
      planHeldMissingModelPull({
        ...base,
        heldReason: "UNKNOWN_MODEL: reported again",
        alreadyOffered,
      }),
    ).toBeNull();
  });

  it("offers nothing for a job with no durable identity to retry", () => {
    for (const jobId of [null, undefined, ""]) {
      expect(planHeldMissingModelPull({ ...base, jobId, heldReason: "UNKNOWN_MODEL" })).toBeNull();
    }
  });
});
