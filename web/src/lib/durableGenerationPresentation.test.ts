import { describe, expect, it } from "vitest";
import type { Job } from "../composables/useGenerateStream";
import { applyDurablePresentation } from "./durableGenerationPresentation";

function job(state: Job["state"] = "running"): Job {
  return {
    state,
    error: null,
    cancelRequested: true,
    holdError: "old hold",
    holdCode: "OLD",
    retryable: true,
    retrying: true,
    settledAt: null,
    workStarted: true,
    previewUrl: "data:image/png;base64,AA==",
    progress: { stage: "Starting", queuePosition: 2 },
  } as unknown as Job;
}

describe("applyDurablePresentation (web)", () => {
  it.each([
    [
      { kind: "waiting", reason: "queued", label: "Queued" },
      {
        workStarted: false,
        holdError: null,
        holdCode: null,
        retryable: false,
        retrying: false,
        progress: { stage: "Queued" },
      },
    ],
    // A resync is a label-only overlay over the hold it covers.
    [
      { kind: "waiting", reason: "resync", label: "Re-syncing" },
      {
        holdError: "old hold",
        holdCode: "OLD",
        progress: { stage: "Re-syncing" },
      },
    ],
    [
      {
        kind: "held",
        label: "Held",
        error: "no model",
        code: "UNKNOWN_MODEL",
        retryable: true,
      },
      {
        workStarted: false,
        holdError: "no model",
        holdCode: "UNKNOWN_MODEL",
        retryable: true,
        progress: { stage: "Held" },
      },
    ],
    [
      { kind: "cancelling", label: "Cancellation pending" },
      { cancelling: true, workStarted: false },
    ],
    [
      { kind: "running", label: "Developing" },
      {
        workStarted: true,
        progress: { stage: "Developing", queuePosition: null },
      },
    ],
    [
      { kind: "cancelled", label: "Cancelled", settledAtMs: 30 },
      {
        state: "canceled",
        cancelling: false,
        cancelRequested: false,
        settledAt: 30,
        previewUrl: null,
      },
    ],
    [
      { kind: "failed", message: "boom", settledAtMs: 31 },
      { state: "error", error: "boom", settledAt: 31, previewUrl: null },
    ],
    [
      { kind: "complete_without_file", message: "no file", settledAtMs: 32 },
      { state: "error", error: "no file", settledAt: 32 },
    ],
    [
      { kind: "rejected", message: "queue full" },
      { state: "error", error: "queue full", settledAt: 50 },
    ],
    // The rail has no "unknown" state; `detached` carries the semantics.
    [
      {
        kind: "unknown",
        label: "Outcome unknown",
        message: "replaced",
        settledAtMs: 50,
      },
      { state: "error", detached: true, error: "replaced", settledAt: 50 },
    ],
  ] as const)("maps %o", (presentation, expected) => {
    const j = job();
    applyDurablePresentation(j, presentation, 50);
    expect(j).toMatchObject(expected);
    if (presentation.kind !== "unknown") expect(j.detached).not.toBe(true);
  });

  it("never rewrites a job that already settled", () => {
    const j = job("done");
    applyDurablePresentation(
      j,
      { kind: "failed", message: "late", settledAtMs: 1 },
      50,
    );
    expect(j).toMatchObject({ state: "done", error: null });
  });
});
