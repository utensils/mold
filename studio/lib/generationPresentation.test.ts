import { describe, expect, it } from "vitest";
import type { GenerationBatchChild } from "../api/generationAdmission";
import {
  createGenerationBatchTracker,
  reduceGenerationLifecycle,
  type GenerationBatchTracker,
  type GenerationLifecycleAction,
} from "./generationLifecycle";
import {
  generationFailureMessage,
  generationTrackerSettled,
  presentGenerationChild,
  presentationIsSettled,
  presentationWorkStarted,
  reconciliationPresentation,
} from "./generationPresentation";

const NOW = 9_000;
const OOM = "ran out of memory. Try a smaller model, image size, or batch.";

function tracker(
  ...actions: GenerationLifecycleAction[]
): GenerationBatchTracker {
  return actions.reduce(
    reduceGenerationLifecycle,
    createGenerationBatchTracker({
      hostId: "host-1",
      expectedInstanceId: "instance-1",
      clientBatchId: "client-1",
      submittedAtMs: 1,
    }),
  );
}

function snapshot(
  state: GenerationBatchChild["state"],
  overrides: Partial<GenerationBatchChild> = {},
): GenerationLifecycleAction {
  return {
    type: "batch_snapshot",
    batch: {
      id: "batch-1",
      client_batch_id: "client-1",
      instance_id: "instance-1",
      durable: true,
      children: [
        {
          index: 1,
          job_id: "job-1",
          state,
          created_at_ms: 10,
          updated_at_ms: 20,
          ...overrides,
        },
      ],
    },
  };
}

const gap = (instanceId: string): GenerationLifecycleAction => ({
  type: "event_gap",
  instanceId,
});

function present(t: GenerationBatchTracker, childIndex = 1) {
  return presentGenerationChild({
    tracker: t,
    childIndex,
    hostLabel: "Render box",
    now: NOW,
  });
}

describe("presentGenerationChild", () => {
  it.each([
    [
      "pending admission",
      tracker(),
      { kind: "waiting", reason: "submitting", label: "Submitting" },
    ],
    [
      "uncertain admission",
      tracker({ type: "admission_uncertain", error: "502" }),
      { kind: "waiting", reason: "confirming", label: "Confirming with host" },
    ],
    [
      "rejected admission",
      tracker({ type: "admission_rejected", error: "queue is full" }),
      { kind: "rejected", message: "queue is full" },
    ],
    [
      "accepted",
      tracker(snapshot("accepted")),
      { kind: "waiting", reason: "queued", label: "Queued" },
    ],
    [
      "queued",
      tracker(snapshot("queued")),
      { kind: "waiting", reason: "queued", label: "Queued" },
    ],
    [
      "paused",
      tracker(snapshot("paused")),
      { kind: "waiting", reason: "paused", label: "Paused after restart" },
    ],
    [
      "running",
      tracker(snapshot("running")),
      { kind: "running", label: "Developing" },
    ],
    [
      "cancelling",
      tracker(snapshot("cancelling")),
      { kind: "cancelling", label: "Cancellation pending" },
    ],
    [
      "held",
      tracker(
        snapshot("held", {
          error: "no model",
          error_code: "UNKNOWN_MODEL",
          retryable: true,
        }),
      ),
      {
        kind: "held",
        label: "Held by host — action required",
        error: "no model",
        code: "UNKNOWN_MODEL",
        retryable: true,
      },
    ],
    [
      "held without a reason",
      tracker(snapshot("held")),
      {
        kind: "held",
        label: "Held by host — action required",
        error: null,
        code: null,
        retryable: false,
      },
    ],
    [
      "complete",
      tracker(
        snapshot("complete", {
          completed_at_ms: 30,
          result: { filename: "p.png", original_filename: "raw.png" },
        }),
      ),
      {
        kind: "complete",
        filename: "p.png",
        originalFilename: "raw.png",
        settledAtMs: 30,
        generationTimeMs: 20,
      },
    ],
    [
      "complete without a file, on the update stamp",
      tracker(snapshot("complete")),
      {
        kind: "complete_without_file",
        message:
          "Render box reported this print complete but published no file.",
        settledAtMs: 20,
      },
    ],
    [
      "failed, composed through the shared describer",
      tracker(
        snapshot("failed", {
          error: "CUDA out of memory",
          completed_at_ms: 25,
        }),
      ),
      { kind: "failed", message: `Render box ${OOM}`, settledAtMs: 25 },
    ],
    [
      "cancelled",
      tracker(snapshot("cancelled", { completed_at_ms: 21 })),
      { kind: "cancelled", label: "Cancelled", settledAtMs: 21 },
    ],
    [
      "a live child behind an event gap",
      tracker(snapshot("running"), gap("instance-1")),
      { kind: "waiting", reason: "resync", label: "Re-syncing with host" },
    ],
    [
      "a live child whose authority was replaced",
      tracker(snapshot("running"), gap("replacement")),
      {
        kind: "unknown",
        label: "Outcome unknown",
        message:
          "Render box was replaced by a new server instance. The previous instance still owns this print's outcome, which is unknown here.",
        settledAtMs: NOW,
      },
    ],
  ] as const)("presents %s", (_name, t, expected) => {
    expect(present(t)).toEqual(expected);
  });

  it("keeps a frozen terminal through a gap or a lost authority", () => {
    expect(
      present(
        tracker(
          snapshot("complete", { result: { filename: "p.png" } }),
          gap("instance-1"),
        ),
      ),
    ).toMatchObject({ kind: "complete" });
    expect(
      present(tracker(snapshot("cancelled"), gap("replacement"))),
    ).toMatchObject({ kind: "cancelled" });
  });

  it("re-syncs a confirmed batch whose snapshot omitted the child", () => {
    expect(present(tracker(snapshot("queued")), 2)).toMatchObject({
      kind: "waiting",
      reason: "resync",
    });
  });
});

describe("reconciliationPresentation", () => {
  it.each([
    ["event_gap", { kind: "resync" }],
    ["incomplete_response", { kind: "resync" }],
    [
      "missing",
      {
        kind: "unknown",
        message:
          "Render box no longer has a record of this print; its outcome is unknown.",
      },
    ],
    [
      "batch_mismatch",
      {
        kind: "unknown",
        message:
          "Render box reported a different batch identity for this print; its outcome is unknown.",
      },
    ],
  ] as const)("maps %s", (reason, expected) => {
    expect(
      reconciliationPresentation({ required: true, reason }, "Render box"),
    ).toEqual(expected);
  });

  it("names the host generically when none is known", () => {
    expect(
      reconciliationPresentation(
        { required: true, reason: "instance_mismatch" },
        null,
      ),
    ).toMatchObject({
      message: expect.stringMatching(/^The host was replaced/),
    });
    expect(
      reconciliationPresentation({ required: false, reason: null }, null),
    ).toEqual({ kind: "none" });
  });
});

describe("generationFailureMessage", () => {
  it.each([
    [
      "the lifecycle's prose",
      { error: " model refused ", terminalError: { message: "x" } },
      "model refused",
    ],
    [
      "a terminal error message",
      { error: null, terminalError: { message: "worker died" } },
      "worker died",
    ],
    [
      "a terminal error field",
      { error: null, terminalError: { error: "typed refusal" } },
      "typed refusal",
    ],
    [
      "an opaque terminal error",
      { error: null, terminalError: { code: 7 } },
      '{"code":7}',
    ],
    ["a default", { error: null, terminalError: null }, "Generation failed"],
    [
      "memory advice",
      { error: "CUDA out of memory", terminalError: null },
      `hal9000 ${OOM}`,
    ],
  ])("prefers %s", (_name, lifecycle, expected) => {
    expect(generationFailureMessage(lifecycle, "hal9000")).toBe(expected);
  });
});

describe("settlement predicates", () => {
  it("splits settled from live presentations", () => {
    for (const state of ["complete", "failed", "cancelled"] as const) {
      expect(presentationIsSettled(present(tracker(snapshot(state))))).toBe(
        true,
      );
    }
    for (const state of [
      "accepted",
      "held",
      "running",
      "cancelling",
    ] as const) {
      expect(presentationIsSettled(present(tracker(snapshot(state))))).toBe(
        false,
      );
    }
    expect(presentationWorkStarted(present(tracker(snapshot("running"))))).toBe(
      true,
    );
    expect(presentationWorkStarted(present(tracker(snapshot("held"))))).toBe(
      false,
    );
  });

  it("settles a tracker on rejection, lost authority, or every child terminal", () => {
    expect(generationTrackerSettled(tracker(), 1)).toBe(false);
    expect(
      generationTrackerSettled(
        tracker({ type: "admission_rejected", error: "no" }),
        1,
      ),
    ).toBe(true);
    expect(
      generationTrackerSettled(
        tracker(snapshot("running"), {
          type: "lookup_missing",
          batchId: "batch-1",
        }),
        1,
      ),
    ).toBe(true);
    expect(
      generationTrackerSettled(
        tracker(snapshot("running"), gap("instance-1")),
        1,
      ),
    ).toBe(false);
    const done = tracker(snapshot("failed"));
    expect(generationTrackerSettled(done, 1)).toBe(true);
    expect(generationTrackerSettled(done, 2)).toBe(false);
    expect(generationTrackerSettled(done, 0)).toBe(false);
  });
});
