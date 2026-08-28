import { describe, expect, it } from "vitest";
import type { QueueEntry, QueuePlan } from "../api/queuePlan";
import {
  QUEUE_BLOCKED_REASONS,
  blockedReasonLabel,
  buildQueueStatusIndex,
  normalizeBlockedReason,
  queuePositionLabel,
  queueStatusFor,
  queueWaitCode,
  queueWaitLabel,
  resolveQueueWait,
} from "./queuePosition";

function entry(id: string, position: number): QueueEntry {
  return {
    id,
    model: "ltx2",
    state: position === 0 ? "running" : "queued",
    started_at_unix_ms: 1_000,
    position,
  };
}

function plan(blocked: Record<string, string>): QueuePlan {
  return {
    plan_version: 1,
    state_version: 1,
    optimizer_state: "settled",
    dirty_since_unix_ms: null,
    next_replan_at_unix_ms: null,
    work_items: Object.entries(blocked).map(([parentId, reason], index) => ({
      work_id: `${parentId}-w`,
      parent_id: parentId,
      work_kind: "generation",
      priority_class: "normal",
      queue_rank: index,
      bypass_count: 0,
      estimate_confidence: "medium",
      blocked_reason: reason,
    })),
  };
}

describe("buildQueueStatusIndex", () => {
  it("projects a host-wide pause onto queued rows without hiding held work", () => {
    const index = buildQueueStatusIndex([
      {
        hostId: "alpha",
        paused: true,
        entries: [
          { id: "waiting", state: "queued", position: 0 } as QueueEntry,
          { id: "parked", state: "held", position: 1 } as QueueEntry,
        ],
      },
    ]);

    expect(
      queueWaitLabel(
        resolveQueueWait(queueStatusFor(index, "alpha", "waiting")),
      ),
    ).toBe("Queue paused");
    expect(
      queueWaitLabel(
        resolveQueueWait(queueStatusFor(index, "alpha", "parked")),
      ),
    ).toBe("Held");
  });

  it("carries the row's lifecycle so a held row resolves as held", () => {
    const index = buildQueueStatusIndex([
      {
        hostId: "alpha",
        entries: [
          { id: "parked", state: "held", position: 0 } as QueueEntry,
          { id: "waiting", state: "queued", position: 0 } as QueueEntry,
        ],
      },
    ]);
    expect(queueStatusFor(index, "alpha", "parked")?.state).toBe("held");
    expect(resolveQueueWait(queueStatusFor(index, "alpha", "parked"))).toEqual({
      kind: "held",
    });
    expect(resolveQueueWait(queueStatusFor(index, "alpha", "waiting"))).toEqual(
      {
        kind: "next",
      },
    );
  });

  it("keys live positions per host so ids from different hosts never collide", () => {
    const index = buildQueueStatusIndex([
      { hostId: "alpha", entries: [entry("job-1", 0), entry("job-2", 1)] },
      { hostId: "beta", entries: [entry("job-1", 3)] },
    ]);
    expect(queueStatusFor(index, "alpha", "job-1")?.position).toBe(0);
    expect(queueStatusFor(index, "alpha", "job-2")?.position).toBe(1);
    expect(queueStatusFor(index, "beta", "job-1")?.position).toBe(3);
  });

  it("returns null for a host that has not been read", () => {
    const index = buildQueueStatusIndex([
      { hostId: "alpha", entries: [entry("job-1", 1)] },
    ]);
    expect(queueStatusFor(index, "gamma", "job-1")).toBeNull();
    expect(queueStatusFor(index, "alpha", "job-9")).toBeNull();
    expect(queueStatusFor(index, "alpha", null)).toBeNull();
  });

  it("tolerates an absent or empty listing without inventing position 0", () => {
    const index = buildQueueStatusIndex([
      { hostId: "alpha", entries: null },
      { hostId: "", entries: [entry("job-1", 1)] },
    ]);
    expect(queueStatusFor(index, "alpha", "job-1")).toBeNull();
    expect(queueStatusFor(index, "", "job-1")).toBeNull();
  });

  it("ignores a non-numeric position rather than rendering NaN", () => {
    const broken = {
      ...entry("job-1", 0),
      position: "second" as unknown as number,
    };
    const index = buildQueueStatusIndex([
      { hostId: "alpha", entries: [broken] },
    ]);
    expect(queueStatusFor(index, "alpha", "job-1")).toEqual({
      state: "running",
      position: null,
      blockedReason: null,
      preparation: null,
    });
  });

  it("joins the plan's blocked reason onto the matching queued job", () => {
    const index = buildQueueStatusIndex([
      {
        hostId: "alpha",
        entries: [entry("job-1", 1)],
        plan: plan({ "job-1": "insufficient_host_ram" }),
      },
    ]);
    expect(queueStatusFor(index, "alpha", "job-1")?.blockedReason).toBe(
      "insufficient_host_ram",
    );
  });

  it("does not report ordinary scheduling bookkeeping as blocked", () => {
    const index = buildQueueStatusIndex([
      {
        hostId: "alpha",
        entries: [entry("job-1", 1)],
        plan: plan({ "job-1": "priority" }),
      },
    ]);
    expect(queueStatusFor(index, "alpha", "job-1")?.blockedReason).toBeNull();
  });

  it("keeps a busy single-GPU host counting the line instead of parking it", () => {
    // `no_idle_device` is what a one-GPU host reports for every job behind the
    // running one (`mold-scheduler/src/planner.rs`) — normal serialization,
    // not a fault, so the row must fall through to its place in line.
    const index = buildQueueStatusIndex([
      {
        hostId: "alpha",
        entries: [entry("job-1", 3)],
        plan: plan({ "job-1": "no_idle_device" }),
      },
    ]);
    const status = queueStatusFor(index, "alpha", "job-1");
    expect(status?.blockedReason).toBeNull();
    expect(queueWaitLabel(resolveQueueWait(status))).toBe("#3 in line");
  });
});

describe("queuePositionLabel", () => {
  it("names the line for jobs actually waiting behind something", () => {
    expect(queuePositionLabel(1)).toBe("#1 in line");
    expect(queuePositionLabel(4)).toBe("#4 in line");
  });

  it("says nothing for the head of the queue or a missing position", () => {
    expect(queuePositionLabel(0)).toBeNull();
    expect(queuePositionLabel(null)).toBeNull();
    expect(queuePositionLabel(undefined)).toBeNull();
    expect(queuePositionLabel(Number.NaN)).toBeNull();
  });
});

describe("blockedReasonLabel", () => {
  it("gives the host-RAM stall plain language", () => {
    expect(blockedReasonLabel("insufficient_host_ram")).toBe(
      "Waiting for memory",
    );
    expect(blockedReasonLabel("insufficient_vram")).toBe(
      "Waiting for GPU memory",
    );
  });

  it("never leaks a raw scheduler string for a reason it does not know", () => {
    expect(blockedReasonLabel("waiting_on_download")).toBe(
      "Waiting on the host",
    );
    expect(normalizeBlockedReason("warm_resident")).toBeNull();
    expect(blockedReasonLabel("warm_resident")).toBeNull();
    expect(blockedReasonLabel(null)).toBeNull();
  });

  it("treats every ordinary planner wait as bookkeeping, not a fault", () => {
    for (const reason of [
      "no_idle_device",
      "lower_priority_opening",
      "warm_wait",
      "dependency_wait",
    ]) {
      expect(blockedReasonLabel(reason)).toBeNull();
      expect(normalizeBlockedReason(reason)).toBeNull();
    }
  });
});

/**
 * One row per `QueueBlockedReason::as_str()` value in
 * `crates/mold-core/src/types.rs`. Every reason must resolve to something a
 * person can read: either its own copy, or silence that lets the row keep
 * counting its place in line.
 */
describe("queue blocked-reason vocabulary", () => {
  it("classifies every reason the server can send", () => {
    for (const reason of QUEUE_BLOCKED_REASONS) {
      const label = blockedReasonLabel(reason);
      if (label === null) continue;
      expect(label, reason).not.toBe(reason);
      expect(label, reason).not.toBe(reason.replaceAll("_", " "));
      expect(label[0], reason).toBe(label[0]?.toUpperCase());
    }
  });

  it("never renders an internal identifier", () => {
    for (const reason of [...QUEUE_BLOCKED_REASONS, "a_brand_new_reason"]) {
      const label = blockedReasonLabel(reason);
      if (label !== null) expect(label, reason).not.toContain("_");
      expect(
        queueWaitLabel(resolveQueueWait({ blockedReason: reason })),
      ).not.toContain("_");
      expect(
        queueWaitCode(resolveQueueWait({ blockedReason: reason })),
      ).not.toContain("_");
    }
  });

  it("gives an unknown future reason generic but honest copy", () => {
    expect(blockedReasonLabel("solar_flare")).toBe("Waiting on the host");
  });

  it("always renders something for a waiting row", () => {
    for (const reason of [...QUEUE_BLOCKED_REASONS, "solar_flare", null]) {
      for (const position of [null, 0, 1, 7]) {
        const wait = resolveQueueWait({ position, blockedReason: reason });
        expect(
          queueWaitLabel(wait).length,
          `${reason}/${position}`,
        ).toBeGreaterThan(0);
        expect(
          queueWaitCode(wait).length,
          `${reason}/${position}`,
        ).toBeGreaterThan(0);
      }
    }
  });
});

describe("resolveQueueWait", () => {
  it("reads a held row as held, whatever position the listing gave it", () => {
    // A held row keeps its traversal index in `GET /api/queue`; index 0 is not
    // "Next up" for work the host will never start on its own.
    expect(resolveQueueWait({ state: "held", position: 0 })).toEqual({
      kind: "held",
    });
    expect(
      resolveQueueWait({
        state: "held",
        position: 1,
        blockedReason: "insufficient_vram",
      }),
    ).toEqual({ kind: "held" });
    expect(
      queueWaitLabel(resolveQueueWait({ state: "held", position: 0 })),
    ).toBe("Held");
    expect(
      queueWaitCode(resolveQueueWait({ state: "held", position: 0 })),
    ).toBe("HELD");
    expect(resolveQueueWait({ state: "queued", position: 0 })).toEqual({
      kind: "next",
    });
  });

  it("reads a restart-paused row as paused instead of in line", () => {
    const wait = resolveQueueWait({ state: "paused", position: 0 });
    expect(wait).toEqual({ kind: "paused" });
    expect(queueWaitLabel(wait)).toBe("Paused after restart");
    expect(queueWaitCode(wait)).toBe("PAUSED");
  });

  it("names the head of the line rather than staying silent", () => {
    expect(resolveQueueWait({ position: 0 })).toEqual({ kind: "next" });
    expect(queueWaitLabel(resolveQueueWait({ position: 0 }))).toBe("Next up");
    expect(queueWaitCode(resolveQueueWait({ position: 0 }))).toBe("NEXT UP");
  });

  it("counts the line for everyone behind it", () => {
    expect(resolveQueueWait({ position: 2 })).toEqual({
      kind: "position",
      position: 2,
    });
    expect(queueWaitLabel(resolveQueueWait({ position: 2 }))).toBe(
      "#2 in line",
    );
    expect(queueWaitCode(resolveQueueWait({ position: 2 }))).toBe("QUEUED #2");
  });

  it("degrades to today's plain pill when the host lists nothing", () => {
    expect(resolveQueueWait(null)).toEqual({ kind: "queued" });
    expect(queueWaitLabel(resolveQueueWait(null))).toBe("Queued");
    expect(queueWaitCode(resolveQueueWait(undefined))).toBe("QUEUED");
  });

  it("lets an actionable reason outrank the position", () => {
    const wait = resolveQueueWait({
      position: 4,
      blockedReason: "insufficient_host_ram",
    });
    expect(wait).toEqual({ kind: "blocked", label: "Waiting for memory" });
    expect(queueWaitCode(wait)).toBe("WAITING FOR MEMORY");
  });

  it("lets a benign reason fall through to the position", () => {
    expect(
      resolveQueueWait({ position: 4, blockedReason: "no_idle_device" }),
    ).toEqual({ kind: "position", position: 4 });
  });
});

describe("preparing", () => {
  it("names a preparing job's phase and progress from the plan", () => {
    const index = buildQueueStatusIndex([
      {
        hostId: "h",
        entries: [{ id: "job" }],
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [
            {
              work_id: "job",
              parent_id: "job",
              work_kind: "generation",
              priority_class: "user",
              queue_rank: 0,
              bypass_count: 0,
              estimate_confidence: "low",
              reason: "not_ready",
              blocked_reason: "preparing",
              preparation_elapsed_ms: 4_200,
              preparation_progress: {
                component: "Verifying MiniMax H3 artifacts",
                bytes_done: 41,
                bytes_total: 100,
              },
            },
          ],
        },
      } as never,
    ]);
    const status = queueStatusFor(index, "h", "job");
    expect(status?.preparation).toEqual({
      component: "Verifying MiniMax H3 artifacts",
      fraction: 0.41,
      elapsedMs: 4_200,
    });
    expect(queueWaitLabel(resolveQueueWait(status))).toBe(
      "Preparing · Verifying MiniMax H3 artifacts 41%",
    );
    expect(queueWaitCode(resolveQueueWait(status))).toBe(
      "PREPARING · VERIFYING MINIMAX H3 ARTIFACTS 41%",
    );
  });

  it("still says Preparing when the preparer reports no progress", () => {
    expect(
      queueWaitLabel(
        resolveQueueWait({ position: 3, blockedReason: "preparing" }),
      ),
    ).toBe("Preparing");
    expect(
      queueWaitLabel(
        resolveQueueWait({
          blockedReason: "preparing",
          preparation: {
            component: "Preparing flux-dev:q8",
            fraction: null,
            elapsedMs: 10,
          },
        }),
      ),
    ).toBe("Preparing · Preparing flux-dev:q8");
  });

  it("outranks the position because the host is already working on it", () => {
    expect(
      resolveQueueWait({ position: 2, blockedReason: "preparing" }),
    ).toEqual({
      kind: "blocked",
      label: "Preparing",
    });
  });
});
