import { describe, expect, it } from "vitest";
import type { QueueEntry, QueuePlan } from "../api/queuePlan";
import {
  blockedReasonLabel,
  buildQueueStatusIndex,
  normalizeBlockedReason,
  queuePositionLabel,
  queueStatusFor,
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
      position: null,
      blockedReason: null,
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

  it("falls back to the normalized reason for anything newer", () => {
    expect(blockedReasonLabel("waiting_on_download")).toBe(
      "waiting on download",
    );
    expect(normalizeBlockedReason("warm_resident")).toBeNull();
    expect(blockedReasonLabel("warm_resident")).toBeNull();
    expect(blockedReasonLabel(null)).toBeNull();
  });
});
