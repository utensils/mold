import { describe, expect, it, vi } from "vitest";
import type { GenerationChildPresentation } from "./generationPresentation";
import {
  applyDurablePresentationWith,
  clearHold,
  type DurableRowHold,
  type DurableRowSurface,
} from "./durableGenerationPresentation";

interface Row extends DurableRowHold {
  live: boolean;
}

function row(over: Partial<Row> = {}): Row {
  return {
    live: true,
    holdError: null,
    holdCode: null,
    retryable: false,
    retrying: false,
    ...over,
  };
}

const held = (over: Partial<Row> = {}) =>
  row({
    holdError: "no machine has it",
    holdCode: "model_missing",
    retryable: true,
    retrying: true,
    ...over,
  });

function surface(): DurableRowSurface<Row> & {
  [K in keyof DurableRowSurface<Row>]: ReturnType<typeof vi.fn>;
} {
  return {
    isLive: vi.fn((job: Row) => job.live),
    waiting: vi.fn(),
    held: vi.fn(),
    cancelling: vi.fn(),
    running: vi.fn(),
    cancelled: vi.fn(),
    failed: vi.fn(),
    rejected: vi.fn(),
    unknown: vi.fn(),
  } as never;
}

const waitingArm = (
  reason: "submitting" | "confirming" | "queued" | "paused" | "resync",
): GenerationChildPresentation => ({
  kind: "waiting",
  reason,
  label: "Waiting",
});

const arm = (p: GenerationChildPresentation) => p;

const P = {
  waiting: waitingArm,
  held: arm({
    kind: "held",
    label: "Held",
    error: "no machine has it",
    code: "model_missing",
    retryable: true,
  }),
  cancelling: arm({ kind: "cancelling", label: "Cancelling" }),
  running: arm({ kind: "running", label: "Rendering" }),
  complete: arm({
    kind: "complete",
    filename: "a.png",
    originalFilename: null,
    settledAtMs: 5,
    generationTimeMs: 1,
  }),
  cancelled: arm({ kind: "cancelled", label: "Cancelled", settledAtMs: 5 }),
  failed: arm({ kind: "failed", message: "boom", settledAtMs: 5 }),
  withoutFile: arm({
    kind: "complete_without_file",
    message: "published no file",
    settledAtMs: 5,
  }),
  rejected: arm({ kind: "rejected", message: "refused" }),
  unknown: arm({
    kind: "unknown",
    label: "Outcome unknown",
    message: "nothing more can arrive",
    settledAtMs: 5,
  }),
};

describe("clearHold", () => {
  it("puts every hold field back, including the retry flag", () => {
    const job = held();
    clearHold(job);
    expect(job).toMatchObject({
      holdError: null,
      holdCode: null,
      retryable: false,
      retrying: false,
    });
  });
});

describe("applyDurablePresentationWith", () => {
  it("never moves a row the surface calls settled", () => {
    const s = surface();
    const job = row({ live: false });
    applyDurablePresentationWith(s)(job, P.failed);
    expect(s.failed).not.toHaveBeenCalled();
  });

  it("hands each arm to its own handler", () => {
    const s = surface();
    const job = row();
    const apply = applyDurablePresentationWith(s);
    apply(job, P.waiting("queued"));
    apply(job, P.held);
    apply(job, P.cancelling);
    apply(job, P.running);
    apply(job, P.cancelled);
    apply(job, P.rejected);
    apply(job, P.unknown);
    for (const arm of [
      "waiting",
      "held",
      "cancelling",
      "running",
      "cancelled",
      "rejected",
      "unknown",
    ] as const) {
      expect(s[arm], arm).toHaveBeenCalledTimes(1);
    }
  });

  /*
   * A completion that names no file is a contradiction the reader must see,
   * and both surfaces have always shown it exactly as they show a failure.
   */
  it("treats a completion with no file as a failure", () => {
    const s = surface();
    const job = row();
    applyDurablePresentationWith(s)(job, P.withoutFile);
    expect(s.failed).toHaveBeenCalledWith(job, P.withoutFile);
  });

  /*
   * The result arm is deliberately unhandled: hydrating the media from what
   * the host published belongs to whoever owns the row, not to a label map.
   */
  it("leaves a finished child entirely alone", () => {
    const s = surface();
    const job = row();
    applyDurablePresentationWith(s)(job, P.complete);
    for (const arm of Object.values(s)) {
      if (arm !== s.isLive) expect(arm).not.toHaveBeenCalled();
    }
  });

  describe("the hold", () => {
    it("is written from the held arm before the surface sees it", () => {
      const s = surface();
      const job = row();
      applyDurablePresentationWith(s)(job, P.held);
      expect(job).toMatchObject({
        holdError: "no machine has it",
        holdCode: "model_missing",
        retryable: true,
      });
    });

    it("is over once the child waits, cancels or runs", () => {
      for (const p of [P.waiting("queued"), P.cancelling, P.running]) {
        const job = held();
        applyDurablePresentationWith(surface())(job, p);
        expect(job.holdError, JSON.stringify(p)).toBeNull();
        expect(job.retrying).toBe(false);
      }
    });

    // A resync is a label-only overlay: the hold it covers is still the
    // machine's word until the snapshot says otherwise.
    it("survives a resync, which only relabels", () => {
      const job = held();
      applyDurablePresentationWith(surface())(job, P.waiting("resync"));
      expect(job.holdError).toBe("no machine has it");
      expect(job.retryable).toBe(true);
    });
  });
});
