import { describe, expect, it } from "vitest";
import { newJob } from "../lib/generationJob";
import {
  createGenerationBatchTracker,
  reduceGenerationLifecycle,
} from "@studio/lib/generationLifecycle";
import type { MobileDurableGenerationRecovery } from "./mobileGenerationRecovery";
import {
  applyMobileDurablePresentation,
  mobileDurableHeld,
  presentMobileDurableChild,
} from "./generationPresentation";

function job() {
  return newJob({ prompt: "a fox", model: "m", width: 8, height: 8, steps: 1, guidance: 1 });
}

function recovery(state: "held" | "queued"): MobileDurableGenerationRecovery {
  const tracker = reduceGenerationLifecycle(
    createGenerationBatchTracker({
      hostId: "host",
      expectedInstanceId: "i",
      clientBatchId: "c",
      submittedAtMs: 1,
    }),
    {
      type: "batch_snapshot",
      batch: {
        id: "b",
        client_batch_id: "c",
        instance_id: "i",
        durable: true,
        children: [
          {
            index: 1,
            job_id: "j",
            state,
            error: "no model",
            error_code: "UNKNOWN_MODEL",
            retryable: true,
            created_at_ms: 1,
            updated_at_ms: 2,
          },
        ],
      },
    },
  );
  return {
    version: 1,
    tracker,
    presentations: [],
    cancelRequestedChildIndexes: [],
    claimedEffects: {},
  };
}

describe("mobile durable presentation", () => {
  it("carries the persisted cancel tap onto live rows only", () => {
    const waiting = job();
    applyMobileDurablePresentation(
      waiting,
      { kind: "waiting", reason: "submitting", label: "Submitting" },
      { cancelRequested: true },
    );
    expect(waiting).toMatchObject({ status: "queued", stage: "Submitting", cancelling: true });
    const cancelled = job();
    cancelled.cancelling = true;
    applyMobileDurablePresentation(
      cancelled,
      { kind: "cancelled", label: "Cancelled", settledAtMs: 5 },
      { cancelRequested: true },
    );
    expect(cancelled).toMatchObject({ status: "error", error: "Cancelled", cancelling: false });
  });

  it("answers the hold from the shared presentation", () => {
    const held = presentMobileDurableChild(recovery("held"), 1, "Studio", 9);
    expect(mobileDurableHeld(held)).toEqual({
      error: "no model",
      code: "UNKNOWN_MODEL",
      retryable: true,
    });
    expect(mobileDurableHeld(presentMobileDurableChild(recovery("queued"), 1, "Studio", 9))).toBe(
      null,
    );
  });
});
