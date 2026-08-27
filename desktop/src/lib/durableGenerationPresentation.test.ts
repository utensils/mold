import { describe, expect, it } from "vitest";
import { newJob } from "./generationJob";
import { applyDurablePresentation } from "./durableGenerationPresentation";

function job() {
  const j = newJob({ prompt: "a fox", model: "m", width: 8, height: 8, steps: 1, guidance: 1 });
  Object.assign(j, {
    holdError: "old hold",
    holdCode: "OLD",
    retryable: true,
    retrying: true,
    cancelling: true,
    interrupted: true,
    previewUrl: "blob:preview",
  });
  return j;
}

describe("applyDurablePresentation (desktop)", () => {
  it.each([
    // A plain queue wait keeps `stage` empty so the live position renders.
    [
      { kind: "waiting", reason: "queued", label: "Queued" },
      {
        status: "queued",
        stage: null,
        holdError: null,
        holdCode: null,
        retryable: false,
        retrying: false,
      },
    ],
    [
      { kind: "waiting", reason: "submitting", label: "Submitting" },
      { status: "queued", stage: "Submitting" },
    ],
    // A resync is a label-only overlay over the hold it covers.
    [
      { kind: "waiting", reason: "resync", label: "Re-syncing" },
      { stage: "Re-syncing", holdError: "old hold", holdCode: "OLD" },
    ],
    [
      { kind: "held", label: "Held", error: "no model", code: "UNKNOWN_MODEL", retryable: true },
      {
        status: "queued",
        stage: "Held",
        holdError: "no model",
        holdCode: "UNKNOWN_MODEL",
        retryable: true,
      },
    ],
    [
      { kind: "cancelling", label: "Cancellation pending" },
      { status: "queued", stage: "Cancellation pending", cancelling: true },
    ],
    [
      { kind: "running", label: "Developing" },
      { status: "loading", stage: "Developing" },
    ],
    [
      { kind: "cancelled", label: "Cancelled", settledAtMs: 30 },
      { status: "error", error: "Cancelled", cancelling: false, settledAtMs: 30, previewUrl: null },
    ],
    [
      { kind: "failed", message: "boom", settledAtMs: 31 },
      { status: "error", error: "boom", settledAtMs: 31 },
    ],
    [
      { kind: "complete_without_file", message: "no file", settledAtMs: 32 },
      { status: "error", error: "no file", settledAtMs: 32 },
    ],
    [
      { kind: "rejected", message: "queue full" },
      { status: "error", error: "queue full" },
    ],
    [
      { kind: "unknown", label: "Outcome unknown", message: "replaced", settledAtMs: 40 },
      {
        status: "error",
        stage: "Outcome unknown",
        error: "replaced",
        interrupted: false,
        retryable: false,
        settledAtMs: 40,
      },
    ],
  ] as const)("maps %o", (presentation, expected) => {
    const j = job();
    applyDurablePresentation(j, presentation);
    expect(j).toMatchObject(expected);
    if (j.status === "error") expect(j.settledAtMs).not.toBeNull();
  });

  it("leaves a settled job and the complete arm alone", () => {
    const done = job();
    done.status = "complete";
    applyDurablePresentation(done, { kind: "failed", message: "late", settledAtMs: 1 });
    expect(done).toMatchObject({ status: "complete", error: null });
    const live = job();
    applyDurablePresentation(live, {
      kind: "complete",
      filename: "p.png",
      originalFilename: null,
      settledAtMs: 1,
      generationTimeMs: 1,
    });
    expect(live.status).toBe("queued");
  });
});
