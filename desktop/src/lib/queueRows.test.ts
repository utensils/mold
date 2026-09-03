import { describe, expect, it } from "vitest";
import { queueSentence, rowGlyph, rowStatusLine, rowTitle, rowTone } from "./queueRows";
import type { QueueRow } from "../composables/useQueueActivity";
import type { Job } from "./generationJob";

function job(part: Partial<Job> = {}): Job {
  return {
    clientId: 1,
    batchId: 1,
    id: "srv-1",
    prompt: "a brass teapot",
    model: "flux-dev:q4",
    width: 1024,
    height: 1024,
    guidance: 3.5,
    visualSeed: "1",
    status: "queued",
    queuePosition: 0,
    step: 0,
    total: 28,
    stage: null,
    chainStageIndex: null,
    chainStageCount: null,
    error: null,
    holdError: null,
    holdCode: null,
    retryable: false,
    retrying: false,
    requestWarnings: [],
    interrupted: false,
    ...part,
  } as Job;
}

const print = (part: Partial<Job> = {}): QueueRow => ({
  key: "print:1",
  createdAtMs: 0,
  kind: "print",
  print: job(part),
});

describe("queue rows speak the lexicon", () => {
  it("titles a print by its words and a clip by its scenes", () => {
    expect(rowTitle(print())).toBe("a brass teapot");
    expect(rowTitle(print({ prompt: "  " }))).toBe("flux-dev:q4");
  });

  it("says Waiting · Being made · Finished as sentences", () => {
    expect(rowStatusLine(print())).toBe("Waiting — next up");
    expect(rowStatusLine(print({ status: "denoising", step: 18 }))).toBe(
      "Adding detail — pass 18 of 28",
    );
    expect(rowStatusLine(print({ status: "complete" }))).toBe("Finished — saved to My images");
    expect(rowStatusLine(print({ status: "error", error: "boom" }))).toBe("Failed — boom");
  });

  it("calls a held print Needs a download first, in the blocked colour, never Failed", () => {
    const held = print({
      status: "error",
      error: "model not found",
      holdError: "flux-dev:q4 is not on studio-rack",
      holdCode: "MODEL_NOT_FOUND",
      retryable: true,
    });
    expect(rowStatusLine(held)).toBe("Needs a download first");
    expect(rowTone(held)).toBe("text-warning");
    expect(rowGlyph(held)).toBe("↓");

    const parked = print({ status: "error", holdError: "no free GPU", retryable: true });
    expect(rowStatusLine(parked)).toBe("Held — no free GPU");
    expect(rowTone(parked)).toBe("text-warning");
    expect(rowGlyph(parked)).toBe("·");
  });

  it("writes the status bar's queue clause", () => {
    expect(queueSentence(0, 0, false)).toBe("nothing waiting");
    expect(queueSentence(1, 3, false)).toBe("1 image being made · 3 waiting");
    expect(queueSentence(2, 0, false)).toBe("2 images being made");
    expect(queueSentence(1, 3, true)).toBe("queue paused · 3 waiting");
  });
});
