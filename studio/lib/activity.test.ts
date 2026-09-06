import { describe, expect, it } from "vitest";
import {
  activityAnnouncement,
  activityCountLabel,
  activityDigestLabel,
  mergeActivity,
  partitionActivity,
  queueStatusLabel,
  withLiveQueueStatus,
  MAX_ATTENTION_ROWS,
  SETTLED_VISIBLE_MS,
  type ActivityJobVM,
  type PrintActivityVM,
} from "./activity";
import {
  compareNewestQueueEntry,
  compareNewestSubmitted,
} from "./activityOrder";
import { buildQueueStatusIndex } from "./queuePosition";

function print(extra: Partial<ActivityJobVM> = {}): ActivityJobVM {
  return {
    kind: "print",
    key: "print:1",
    hostId: "local",
    hostLabel: "This device",
    model: "flux-dev:q8",
    prompt: "a cat",
    phase: "queued",
    progress: null,
    chain: null,
    actions: ["cancel"],
    createdAtMs: 100,
    settledAtMs: null,
    ...extra,
  } as ActivityJobVM;
}

describe("compareNewestSubmitted", () => {
  it("sorts newest-first and preserves sending order for equal timestamps", () => {
    const rows = [
      { id: "equal-first", createdAtMs: 20 },
      { id: "oldest", createdAtMs: 10 },
      { id: "equal-second", createdAtMs: 20 },
      { id: "newest", createdAtMs: 30 },
    ];
    expect(rows.sort(compareNewestSubmitted).map(({ id }) => id)).toEqual([
      "newest",
      "equal-first",
      "equal-second",
      "oldest",
    ]);
  });

  it("sorts queue entries newest-first without using scheduler position", () => {
    const entries = [
      { id: "sent-first", started_at_unix_ms: 20, position: 9 },
      { id: "oldest", started_at_unix_ms: 10, position: 0 },
      { id: "sent-second", started_at_unix_ms: 20, position: 1 },
      { id: "newest", started_at_unix_ms: 30, position: 7 },
    ];
    expect(entries.sort(compareNewestQueueEntry).map(({ id }) => id)).toEqual([
      "newest",
      "sent-first",
      "sent-second",
      "oldest",
    ]);
  });
});

describe("mergeActivity", () => {
  it("keeps active work newest-first regardless of phase, then settled by recency", () => {
    const rows = mergeActivity([
      print({ key: "print:1", phase: "done", createdAtMs: 400 }),
      print({ key: "print:2", phase: "running", createdAtMs: 100 }),
      print({ key: "print:3", phase: "queued", createdAtMs: 300 }),
      print({ key: "print:4", phase: "done", createdAtMs: 500 }),
    ]);
    expect(rows.map((vm) => vm.key)).toEqual([
      "print:3", // newer queued work stays above older running work
      "print:2",
      "print:4", // then settled, newest first
      "print:1",
    ]);
  });

  // Regression: print VMs used to carry `createdAtMs = job.clientId` (a 1,2,3…
  // counter). Harmless while every print VM was filtered back out after the
  // merge, fatal the moment a failed print keeps a row — a counter always
  // loses to a ~1.7e12 epoch stamp and sorts to the bottom forever.
  it("orders rows on a real wall clock, not a client counter", () => {
    const rows = mergeActivity([
      print({
        key: "print:old",
        phase: "failed",
        createdAtMs: 1_700_000_000_000,
      }),
      print({
        key: "print:new",
        phase: "failed",
        createdAtMs: 1_700_000_100_000,
      }),
    ]);
    expect(rows.map((vm) => vm.key)).toEqual(["print:new", "print:old"]);
  });

  // A long clip the host renders as chained clips is ONE print carrying a
  // stage counter — never a second row, and never a second kind of work.
  it("renders an auto-chained long video as a single print row", () => {
    const rows = mergeActivity([
      print({
        key: "print:long",
        phase: "running",
        chain: { stageIndex: 1, stageCount: 3 },
      }),
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]!.chain).toEqual({ stageIndex: 1, stageCount: 3 });
  });
});

describe("partitionActivity", () => {
  const NOW = 1_700_000_000_000;
  const failed = (key: string, settledAtMs: number) =>
    print({ key, phase: "failed", createdAtMs: NOW, settledAtMs });

  it("puts queued and running rows in active and never in attention", () => {
    const part = partitionActivity(
      [
        print({ key: "print:q", phase: "queued" }),
        print({ key: "print:r", phase: "running" }),
      ],
      { nowMs: NOW },
    );
    expect(part.active.map((vm) => vm.key)).toEqual(["print:q", "print:r"]);
    expect(part.attention).toEqual([]);
  });

  it("renders no row for a finished print", () => {
    const part = partitionActivity(
      [print({ key: "print:d", phase: "done", settledAtMs: NOW - 1_000 })],
      { nowMs: NOW },
    );
    expect(part.active).toEqual([]);
    expect(part.attention).toEqual([]);
  });

  it("never gives a cancelled print an attention row", () => {
    const part = partitionActivity(
      [print({ key: "print:x", phase: "cancelled", settledAtMs: NOW - 1_000 })],
      { nowMs: NOW },
    );
    expect(part.attention).toEqual([]);
  });

  it("keeps a fresh failure and ages out a stale one", () => {
    const fresh = partitionActivity([failed("print:f", NOW - 60_000)], {
      nowMs: NOW,
    });
    expect(fresh.attention.map((vm) => vm.key)).toEqual(["print:f"]);

    const stale = partitionActivity(
      [failed("print:f", NOW - SETTLED_VISIBLE_MS - 1)],
      { nowMs: NOW },
    );
    expect(stale.attention).toEqual([]);
  });

  it("keeps a failure it cannot date, rather than hiding its only pointer", () => {
    const part = partitionActivity(
      [print({ key: "print:f", phase: "failed", settledAtMs: null })],
      { nowMs: NOW },
    );
    expect(part.attention.map((vm) => vm.key)).toEqual(["print:f"]);
  });

  it("keeps the newest rows at the cap and reports the overflow", () => {
    const part = partitionActivity(
      [
        failed("print:a", NOW - 3_000),
        failed("print:b", NOW - 2_000),
        failed("print:c", NOW - 1_000),
      ],
      { nowMs: NOW, maxAttentionRows: MAX_ATTENTION_ROWS },
    );
    expect(part.attention.map((vm) => vm.key)).toEqual(["print:c", "print:b"]);
    expect(part.hiddenAttention).toBe(1);
  });

  it("drops dismissed rows without counting them as hidden", () => {
    const rows = [failed("print:f", NOW - 1_000)];
    expect(partitionActivity(rows, { nowMs: NOW }).attention).toHaveLength(1);
    const dropped = partitionActivity(rows, {
      nowMs: NOW,
      dismissed: ["print:f"],
    });
    expect(dropped.attention).toEqual([]);
    expect(dropped.hiddenAttention).toBe(0);
  });
});

describe("activityDigestLabel", () => {
  it("stays silent when the strip is showing everything", () => {
    expect(activityDigestLabel({ hiddenAttention: 0 })).toBeNull();
  });

  it("counts the failures the cap held back", () => {
    expect(activityDigestLabel({ hiddenAttention: 1 })).toBe("1 failed");
    expect(activityDigestLabel({ hiddenAttention: 3 })).toBe("3 failed");
  });
});

describe("withLiveQueueStatus", () => {
  const index = buildQueueStatusIndex([
    {
      hostId: "local",
      entries: [
        {
          id: "srv-prep",
          model: "m",
          state: "queued",
          started_at_unix_ms: 2,
          position: 1,
        },
        {
          id: "srv-run",
          model: "m",
          state: "running",
          started_at_unix_ms: 1,
          position: 0,
        },
        {
          id: "srv-2",
          model: "m",
          state: "queued",
          started_at_unix_ms: 2,
          position: 2,
        },
        {
          id: "srv-3",
          model: "m",
          state: "queued",
          started_at_unix_ms: 3,
          position: 3,
        },
      ],
      plan: {
        plan_version: 1,
        state_version: 1,
        optimizer_state: "settled",
        dirty_since_unix_ms: null,
        next_replan_at_unix_ms: null,
        work_items: [
          {
            work_id: "w-prep",
            parent_id: "srv-prep",
            work_kind: "generation",
            priority_class: "normal",
            queue_rank: 2,
            bypass_count: 0,
            estimate_confidence: "low",
            blocked_reason: "preparing",
            preparation_elapsed_ms: 4_200,
            preparation_progress: {
              component: "Verifying model files",
              bytes_done: 27,
              bytes_total: 100,
            },
          },
          {
            work_id: "w3",
            parent_id: "srv-3",
            work_kind: "generation",
            priority_class: "normal",
            queue_rank: 3,
            bypass_count: 0,
            estimate_confidence: "low",
            blocked_reason: "insufficient_host_ram",
          },
        ],
      },
    },
  ]);

  it("maps a queued job to its live position", () => {
    const vm = withLiveQueueStatus(print() as PrintActivityVM, index, "srv-2");
    expect(vm.queuePosition).toBe(2);
    expect(queueStatusLabel(vm)).toBe("#2 in line");
  });

  it("leaves the position absent when the host never listed the job", () => {
    const vm = withLiveQueueStatus(
      print() as PrintActivityVM,
      index,
      "srv-missing",
    );
    expect(vm.queuePosition).toBeUndefined();
    // No evidence still says the one true thing: it is queued.
    expect(queueStatusLabel(vm)).toBe("Queued");
    const unread = withLiveQueueStatus(
      print() as PrintActivityVM,
      null,
      "srv-2",
    );
    expect(unread.queuePosition).toBeUndefined();
    const unsubmitted = withLiveQueueStatus(
      print() as PrintActivityVM,
      index,
      "",
    );
    expect(unsubmitted.queuePosition).toBeUndefined();
  });

  it("never carries a position on running or settled rows", () => {
    for (const phase of ["running", "done", "failed", "cancelled"] as const) {
      const vm = withLiveQueueStatus(
        print({
          phase,
          settledAtMs: phase === "running" ? null : 10,
        }) as PrintActivityVM,
        index,
        "srv-2",
      );
      expect(vm.queuePosition).toBeUndefined();
      expect(queueStatusLabel(vm)).toBeNull();
    }
  });

  it("says why a parked job is waiting instead of counting its place", () => {
    const vm = withLiveQueueStatus(print() as PrintActivityVM, index, "srv-3");
    expect(vm.blockedReason).toBe("insufficient_host_ram");
    expect(queueStatusLabel(vm)).toBe("Waiting for memory");
  });

  it("carries live preparation detail into the shared activity label", () => {
    const vm = withLiveQueueStatus(
      print() as PrintActivityVM,
      index,
      "srv-prep",
    );
    expect(vm.preparation).toEqual({
      component: "Verifying model files",
      fraction: 0.27,
      elapsedMs: 4_200,
    });
    expect(queueStatusLabel(vm)).toBe("Preparing · Verifying model files 27%");
  });

  it("names the job at the head of the queue", () => {
    const vm = withLiveQueueStatus(
      print() as PrintActivityVM,
      index,
      "srv-run",
    );
    expect(vm.queuePosition).toBe(0);
    expect(queueStatusLabel(vm)).toBe("Next up");
  });

  it("says nothing for work that is not waiting", () => {
    expect(queueStatusLabel(print({ phase: "running" }))).toBeNull();
    expect(queueStatusLabel(print({ phase: "done" }))).toBeNull();
  });
});

describe("activity counts", () => {
  it("never calls queued work active", () => {
    // The iPhone screenshot: one z-image job running, four behind it.
    expect(activityCountLabel({ running: 1, waiting: 4 })).toBe(
      "1 active · 4 queued",
    );
    expect(activityAnnouncement({ running: 1, waiting: 4 })).toBe(
      "1 active generation, 4 queued.",
    );
  });

  it("drops the half that is zero", () => {
    expect(activityCountLabel({ running: 2, waiting: 0 })).toBe("2 active");
    expect(activityCountLabel({ running: 0, waiting: 3 })).toBe("3 queued");
    expect(activityAnnouncement({ running: 1, waiting: 0 })).toBe(
      "1 active generation.",
    );
    expect(activityAnnouncement({ running: 0, waiting: 1 })).toBe(
      "1 queued generation.",
    );
    expect(activityAnnouncement({ running: 0, waiting: 3 })).toBe(
      "3 queued generations.",
    );
  });

  it("stays honest when only settled rows are left on screen", () => {
    expect(activityCountLabel({ running: 0, waiting: 0 })).toBe("0 active");
    expect(activityAnnouncement({ running: 0, waiting: 0 })).toBe(
      "No active generations.",
    );
  });
});
