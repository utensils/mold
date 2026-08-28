import { describe, expect, it } from "vitest";
import {
  hostMemoryLevel,
  hostMemoryScheduleLabel,
  parseHostMemory,
} from "./hostMemory";

const snapshot = {
  total_bytes: 64_000_000_000,
  available_bytes: 40_000_000_000,
  headroom_bytes: 32_000_000_000,
  safety_floor_bytes: 4_000_000_000,
};

describe("parseHostMemory", () => {
  it("reads a complete additive snapshot", () => {
    expect(parseHostMemory(snapshot)).toEqual(snapshot);
  });

  it("reads a partial or malformed payload as absent", () => {
    expect(parseHostMemory(undefined)).toBeNull();
    expect(parseHostMemory(null)).toBeNull();
    expect(parseHostMemory([])).toBeNull();
    expect(parseHostMemory({ ...snapshot, headroom_bytes: "lots" })).toBeNull();
    expect(
      parseHostMemory({ ...snapshot, safety_floor_bytes: NaN }),
    ).toBeNull();
    const { total_bytes: _dropped, ...missing } = snapshot;
    expect(parseHostMemory(missing)).toBeNull();
  });

  it("reads the additive ZFS ARC credit and tolerates its absence or garbage", () => {
    expect(
      parseHostMemory({
        ...snapshot,
        reclaimable_zfs_arc_bytes: 14_000_000_000,
      }),
    ).toEqual({ ...snapshot, reclaimable_zfs_arc_bytes: 14_000_000_000 });
    expect(
      parseHostMemory({ ...snapshot, reclaimable_zfs_arc_bytes: 0 }),
    ).toEqual({ ...snapshot, reclaimable_zfs_arc_bytes: 0 });
    // An older server omits it; a broken one must not spoil the meter.
    expect(parseHostMemory(snapshot)).not.toHaveProperty(
      "reclaimable_zfs_arc_bytes",
    );
    expect(
      parseHostMemory({ ...snapshot, reclaimable_zfs_arc_bytes: "lots" }),
    ).toEqual(snapshot);
    expect(
      parseHostMemory({ ...snapshot, reclaimable_zfs_arc_bytes: null }),
    ).toEqual(snapshot);
  });
});

describe("hostMemoryScheduleLabel", () => {
  const fmt = (bytes: number) => `${(bytes / 1_000_000_000).toFixed(1)} GB`;

  it("names the credit only when positive", () => {
    expect(hostMemoryScheduleLabel(snapshot, fmt)).toBe(
      "32.0 GB available to schedule",
    );
    expect(hostMemoryScheduleLabel(snapshot, fmt, { withTotal: true })).toBe(
      "32.0 GB of 64.0 GB available to schedule",
    );
    expect(
      hostMemoryScheduleLabel(
        { ...snapshot, reclaimable_zfs_arc_bytes: 0 },
        fmt,
      ),
    ).toBe("32.0 GB available to schedule");
    expect(
      hostMemoryScheduleLabel(
        { ...snapshot, reclaimable_zfs_arc_bytes: 14_000_000_000 },
        fmt,
      ),
    ).toBe(
      "32.0 GB available to schedule (includes 14.0 GB evictable ZFS ARC)",
    );
    expect(
      hostMemoryScheduleLabel(
        { ...snapshot, reclaimable_zfs_arc_bytes: 14_000_000_000 },
        fmt,
        { withTotal: true },
      ),
    ).toBe(
      "32.0 GB of 64.0 GB available to schedule (includes 14.0 GB evictable ZFS ARC)",
    );
  });
});

describe("hostMemoryLevel", () => {
  it("stays silent when the host does not report host memory", () => {
    expect(hostMemoryLevel(null)).toBeNull();
    expect(hostMemoryLevel(undefined)).toBeNull();
  });

  it("is ok while headroom clears the safety floor", () => {
    expect(hostMemoryLevel(snapshot)).toBe("ok");
  });

  it("warns within one safety floor of the wall", () => {
    expect(
      hostMemoryLevel({ ...snapshot, headroom_bytes: 3_000_000_000 }),
    ).toBe("warn");
  });

  it("is critical once nothing is spendable", () => {
    expect(hostMemoryLevel({ ...snapshot, headroom_bytes: 0 })).toBe(
      "critical",
    );
    expect(hostMemoryLevel({ ...snapshot, headroom_bytes: -1 })).toBe(
      "critical",
    );
  });

  it("treats a zero safety floor as no warn band rather than constant warning", () => {
    expect(
      hostMemoryLevel({
        ...snapshot,
        headroom_bytes: 1,
        safety_floor_bytes: 0,
      }),
    ).toBe("ok");
  });
});
