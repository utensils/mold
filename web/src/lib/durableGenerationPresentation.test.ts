import { describe, expect, it } from "vitest";
import { applyDurablePresentation } from "./durableGenerationPresentation";

function job() {
  return {
    state: "running",
    settledAt: null,
    cancelling: false,
    cancelRequested: false,
    previewUrl: null,
    holdError: null,
    holdCode: null,
    retryable: false,
    retrying: false,
    workStarted: true,
    progress: { stage: "Loading model", queuePosition: null },
  };
}

describe("web durable generation presentation", () => {
  it("keeps the host's running sub-stage across lifecycle reconciliation", () => {
    const current = job();

    applyDurablePresentation(
      current as never,
      { kind: "running", label: "Developing" },
      1,
    );

    expect(current.progress.stage).toBe("Loading model");
  });
});
