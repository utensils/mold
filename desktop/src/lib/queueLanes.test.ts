import { describe, expect, it } from "vitest";
import {
  computeQueueLanes,
  laneForEntry,
  resolveDropAction,
  type LaneEntryLike,
} from "./queueLanes";

function entry(over: Partial<LaneEntryLike> & { id: string }): LaneEntryLike {
  return { state: "queued", position: 0, started_at_unix_ms: 0, ...over };
}

describe("laneForEntry", () => {
  it("uses the running GPU for running rows and target_gpu for queued rows", () => {
    expect(laneForEntry(entry({ id: "a", state: "running", gpu: 1 }))).toBe(1);
    expect(laneForEntry(entry({ id: "b", state: "queued", target_gpu: 2 }))).toBe(2);
  });

  it("returns null (Auto) when the row names no GPU", () => {
    expect(laneForEntry(entry({ id: "a", state: "queued" }))).toBeNull();
    expect(laneForEntry(entry({ id: "b", state: "running" }))).toBeNull();
  });
});

describe("held rows", () => {
  it("keeps held work visible in newest-first display order", () => {
    const rows = [
      entry({ id: "queued", state: "queued", position: 2, started_at_unix_ms: 1 }),
      entry({ id: "run", state: "running", position: 1, started_at_unix_ms: 2 }),
      entry({ id: "held", state: "held", position: 0, started_at_unix_ms: 3 }),
    ];

    const lanes = computeQueueLanes(rows, []);

    expect(lanes[0]!.entries.map((row) => row.id)).toEqual(["held", "run", "queued"]);
  });

  it("puts a held row in the Auto lane rather than claiming a GPU", () => {
    // It is not running anywhere and is not waiting for a turn.
    expect(laneForEntry(entry({ id: "h", state: "held" }))).toBeNull();
  });
});

describe("computeQueueLanes", () => {
  it("keeps a single flat lane when the host reports fewer than two GPUs", () => {
    const rows = [entry({ id: "a" }), entry({ id: "b", position: 1 })];
    for (const ordinals of [[], [0]]) {
      const lanes = computeQueueLanes(rows, ordinals);
      expect(lanes).toHaveLength(1);
      expect(lanes[0]).toMatchObject({ key: "queue" });
      expect(lanes[0]!.entries.map((e) => e.id)).toEqual(["a", "b"]);
    }
  });

  it("groups rows into per-GPU lanes by target_gpu / running gpu", () => {
    const rows = [
      entry({ id: "run0", state: "running", gpu: 0 }),
      entry({ id: "q1", target_gpu: 1, position: 1 }),
      entry({ id: "q0", target_gpu: 0, position: 2 }),
    ];
    const lanes = computeQueueLanes(rows, [0, 1]);
    expect(lanes.map((l) => l.key)).toEqual([0, 1]);
    expect(lanes.map((l) => l.label)).toEqual(["GPU 0", "GPU 1"]);
    expect(lanes[0]!.entries.map((e) => e.id)).toEqual(["run0", "q0"]);
    expect(lanes[1]!.entries.map((e) => e.id)).toEqual(["q1"]);
  });

  it("puts Auto rows (no target_gpu) in the default lowest-ordinal lane", () => {
    const rows = [entry({ id: "auto" })];
    const lanes = computeQueueLanes(rows, [0, 1]);
    expect(lanes[0]!.entries.map((e) => e.id)).toEqual(["auto"]);
    expect(lanes[1]!.entries).toEqual([]);
  });

  it("gives an unknown target_gpu its own lane (union with host-reported ordinals)", () => {
    // Host status only knows GPU 0, but a row already targets GPU 1: GPU 1 is a
    // real lane of its own, not folded into Auto/the default lane.
    const rows = [entry({ id: "q1", target_gpu: 1 })];
    const lanes = computeQueueLanes(rows, [0]);
    expect(lanes.map((l) => l.key)).toEqual([0, 1]);
    expect(lanes[0]!.entries).toEqual([]);
    expect(lanes[1]!.entries.map((e) => e.id)).toEqual(["q1"]);
  });

  it("orders each lane newest-first without changing queue positions", () => {
    const rows = [
      entry({ id: "q-late", target_gpu: 0, position: 5, started_at_unix_ms: 3 }),
      entry({ id: "run", state: "running", gpu: 0, position: 9, started_at_unix_ms: 1 }),
      entry({ id: "q-early", target_gpu: 0, position: 1, started_at_unix_ms: 2 }),
    ];
    const lanes = computeQueueLanes(rows, [0, 1]);
    expect(lanes[0]!.entries.map((e) => e.id)).toEqual(["q-late", "q-early", "run"]);
  });
});

describe("resolveDropAction", () => {
  const dragged = { hostId: "hal", entryId: "srv-2", lane: 0 };

  it("reassigns within the owning host", () => {
    expect(resolveDropAction(dragged, "hal", 1)).toEqual({
      kind: "reassign",
      hostId: "hal",
      entryId: "srv-2",
      targetGpu: 1,
    });
  });

  it("rejects cross-host drops — jobs cannot migrate between servers", () => {
    expect(resolveDropAction(dragged, "local", 1)).toEqual({ kind: "reject-cross-host" });
  });

  it("is a no-op without a drag, onto the flat queue lane, or onto the current lane", () => {
    expect(resolveDropAction(null, "hal", 1)).toEqual({ kind: "none" });
    expect(resolveDropAction(dragged, "hal", "queue")).toEqual({ kind: "none" });
    expect(resolveDropAction(dragged, "hal", 0)).toEqual({ kind: "none" });
  });
});
