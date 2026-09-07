import { describe, expect, it } from "vitest";
import {
  batchPositionLabel,
  madeTodayCount,
  queueSentence,
  railStatusLine,
  rowGlyph,
  rowProgressFraction,
  rowStatusLine,
  rowTitle,
  rowTone,
  type QueueRowContext,
} from "./queueRows";
import type { QueueStatus } from "@studio/lib/queuePosition";
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

/** A row the host reports through `/api/activity` rather than one this
 * client is tracking itself — every fleet row, and every row after a
 * reconnect. */
const shared = (part: Record<string, unknown> = {}): QueueRow =>
  ({
    key: "shared:1",
    createdAtMs: 0,
    kind: "shared",
    shared: {
      id: "job-1",
      kind: "generation",
      phase: "running",
      model: "hunyuan3d-2.1:fp16",
      created_at_unix_ms: 0,
      updated_at_unix_ms: 0,
      can_cancel: true,
      ...part,
    },
  }) as unknown as QueueRow;

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
    expect(rowTone(held)).toBe("text-state-blocked");
    expect(rowGlyph(held)).toBe("↓");

    const parked = print({ status: "error", holdError: "no free GPU", retryable: true });
    expect(rowStatusLine(parked)).toBe("Held — no free GPU");
    expect(rowTone(parked)).toBe("text-state-blocked");
    expect(rowGlyph(parked)).toBe("·");
  });

  it("says how long is left only where the host predicted a finish", () => {
    const making = print({ status: "denoising", step: 18 });
    expect(rowStatusLine(making, { etaSeconds: 12 })).toBe("Adding detail — about 12s left");
    expect(rowStatusLine(making, { etaSeconds: null })).toBe("Adding detail — pass 18 of 28");
  });

  it("says what a parked row is parked on, not a bare Waiting", () => {
    const waiting = print({ status: "queued", queuePosition: 2 });
    const wait = (part: Partial<QueueStatus>): QueueRowContext => ({
      wait: {
        state: "queued",
        position: 2,
        blockedReason: null,
        preparation: null,
        explicitlyPaused: null,
        ...part,
      },
    });
    expect(rowStatusLine(waiting)).toBe("Waiting — #2 in line");
    expect(rowStatusLine(waiting, wait({ state: "held" }))).toBe("Held");
    expect(
      rowStatusLine(
        waiting,
        wait({
          blockedReason: "preparing",
          preparation: { component: "flux weights", fraction: 0.42, elapsedMs: null },
        }),
      ),
    ).toBe("Getting a style ready · 42%");
    expect(rowStatusLine(waiting, wait({ blockedReason: "model_not_installed" }))).toBe(
      "Waiting — model not installed",
    );
  });

  /*
   * A row someone paused and a whole queue parked by a restart both arrive as
   * `state: "paused"`. Saying "after restart" for the first is what made
   * pausing ONE job read as the entire queue stopping.
   */
  it("tells a job someone paused apart from a queue parked by a restart", () => {
    const waiting = print({ status: "queued", queuePosition: 2 });
    const paused = (explicitlyPaused: boolean | null): QueueRowContext => ({
      wait: {
        state: "paused",
        position: 2,
        blockedReason: null,
        preparation: null,
        explicitlyPaused,
      },
    });
    expect(rowStatusLine(waiting, paused(true))).toBe("Paused");
    expect(rowStatusLine(waiting, paused(false))).toBe("Paused after restart");
    // A host too old to distinguish them only ever parked at restart, so its
    // rows keep the sentence they have always had.
    expect(rowStatusLine(waiting, paused(null))).toBe("Paused after restart");
  });

  /** `POST /api/queue/pause` holds DISPATCH; the job already on the GPU
   *  finishes (`queue.rs`: "in-flight worker jobs continue"). Calling the
   *  print "paused" told the user the GPU had stopped when it had not. The
   *  pause is the queue's word, said once on the waiting rows and the header. */
  it("keeps a print being made honest under a paused queue", () => {
    expect(rowStatusLine(print({ status: "denoising", step: 18 }), { queuePaused: true })).toBe(
      "Adding detail — pass 18 of 28",
    );
    expect(
      rowStatusLine(print({ status: "denoising", step: 18 }), {
        queuePaused: true,
        etaSeconds: 30,
      }),
    ).toBe("Adding detail — about 30s left");
  });

  it("shortens the rail's dash to a middot while the full view keeps it", () => {
    const making = print({ status: "denoising", step: 18 });
    expect(railStatusLine(making)).toBe("Adding detail · pass 18 of 28");
    expect(rowStatusLine(making)).toBe("Adding detail — pass 18 of 28");
  });

  it("states a batch sibling's place, and says nothing for a lone print", () => {
    const jobs = [1, 2, 3, 4].map((clientId) => job({ clientId, batchId: 7 }));
    expect(batchPositionLabel(print({ clientId: 2, batchId: 7 }), jobs)).toBe("image 2 of 4");
    expect(
      batchPositionLabel(print({ clientId: 1, batchId: 9 }), [job({ batchId: 9 })]),
    ).toBeNull();
  });

  it("counts today's prints from midnight, not from the session", () => {
    const now = new Date("2026-09-03T14:00:00");
    const at = (iso: string) => ({ item: { timestamp: new Date(iso).getTime() / 1000 } });
    expect(
      madeTodayCount(
        [at("2026-09-03T13:59:00"), at("2026-09-03T00:00:00"), at("2026-09-02T23:59:59")],
        now,
      ),
    ).toBe(2);
  });

  it("writes the status bar's queue clause", () => {
    expect(queueSentence(0, 0, false)).toBe("nothing waiting");
    expect(queueSentence(1, 3, false)).toBe("1 image being made · 3 waiting");
    expect(queueSentence(2, 0, false)).toBe("2 images being made");
    expect(queueSentence(1, 3, true)).toBe("queue paused · 3 waiting");
  });
});

/*
 * The meter and the sentence beside it read ONE counter.
 *
 * `rowStatusLine` gave a shared row "Generating PBR views · 11/15" from the
 * host's own `current`/`total` while both meters understood only a print
 * row's denoise step, so the bar fell back to a hard-coded stub and drew ~8%
 * next to a caption saying 73%.
 */
describe("a queue row's meter agrees with its sentence", () => {
  it("measures a shared row by the same counter its caption states", () => {
    const row = shared({ stage: "Generating PBR views", current: 11, total: 15 });
    expect(rowStatusLine(row)).toBe("Generating PBR views · 11/15");
    expect(rowProgressFraction(row)).toBeCloseTo(11 / 15, 5);
  });

  it("measures a print row by its denoise pass", () => {
    expect(rowProgressFraction(print({ status: "denoising", step: 7, total: 28 }))).toBeCloseTo(
      0.25,
      5,
    );
  });

  it("measures nothing it cannot measure, rather than guessing", () => {
    expect(rowProgressFraction(print())).toBeNull();
    expect(rowProgressFraction(print({ status: "loading" }))).toBeNull();
    expect(rowProgressFraction(print({ status: "denoising", total: 0 }))).toBeNull();
    // A host that names a stage without counting it still gets a caption.
    const uncounted = shared({ stage: "Unwrapping mesh" });
    expect(rowStatusLine(uncounted)).toBe("Unwrapping mesh");
    expect(rowProgressFraction(uncounted)).toBeNull();
    expect(rowProgressFraction(shared({ current: 3, total: 0 }))).toBeNull();
  });

  it("never leaves the meter outside its track", () => {
    // A host that over-counts its own stage must not draw past the end.
    expect(rowProgressFraction(shared({ current: 40, total: 15 }))).toBe(1);
    expect(rowProgressFraction(shared({ current: -2, total: 15 }))).toBe(0);
  });
});
