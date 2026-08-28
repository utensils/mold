import { describe, expect, it } from "vitest";
import { planHeldMissingModelPull } from "./missingModelHold";

const base = {
  jobId: "job-1",
  model: "z-image-turbo:q6",
  alreadyOffered: new Set<string>(),
};

describe("planHeldMissingModelPull", () => {
  it("offers the pull for a child the machine parked because the model is missing", () => {
    for (const heldCode of ["UNKNOWN_MODEL", "MODEL_NOT_FOUND"]) {
      expect(planHeldMissingModelPull({ ...base, heldCode })).toEqual({
        model: "z-image-turbo:q6",
        jobId: "job-1",
      });
    }
  });

  it("offers nothing for a hold that is not about the model", () => {
    for (const heldCode of [
      null,
      "",
      // A sentence is never a code, even one that mentions the model.
      "deferred generation preparation failed: model 'z-image-turbo:q6' is not downloaded",
      "QUEUE_PAUSED",
      "DEVICES_DISABLED",
    ]) {
      expect(planHeldMissingModelPull({ ...base, heldCode })).toBeNull();
    }
  });

  /** A machine re-reports the same hold on every reconciliation wave. */
  it("offers a parked print exactly once", () => {
    const alreadyOffered = new Set<string>();
    const first = planHeldMissingModelPull({
      ...base,
      heldCode: "UNKNOWN_MODEL",
      alreadyOffered,
    });
    expect(first).not.toBeNull();
    alreadyOffered.add(first!.jobId);
    expect(
      planHeldMissingModelPull({
        ...base,
        heldCode: "UNKNOWN_MODEL",
        alreadyOffered,
      }),
    ).toBeNull();
  });

  it("offers nothing for a job with no durable identity to retry", () => {
    for (const jobId of [null, undefined, ""]) {
      expect(planHeldMissingModelPull({ ...base, jobId, heldCode: "UNKNOWN_MODEL" })).toBeNull();
    }
  });
});
