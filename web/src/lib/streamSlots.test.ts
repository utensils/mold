import { describe, expect, it, vi } from "vitest";
import { TargetStreamSlots } from "./streamSlots";

describe("TargetStreamSlots", () => {
  it("starts at most four browser streams and promotes waiters in order", () => {
    const pool = new TargetStreamSlots(4);
    const started: number[] = [];
    const releases: Array<() => void> = [];
    for (let i = 0; i < 6; i += 1) {
      pool.schedule("origin", new AbortController().signal, (release) => {
        started.push(i);
        releases[i] = release;
      });
    }
    expect(started).toEqual([0, 1, 2, 3]);
    releases[0]!();
    expect(started).toEqual([0, 1, 2, 3, 4]);
    releases[1]!();
    expect(started).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it("removes an aborted waiter without consuming a slot", () => {
    const pool = new TargetStreamSlots(1);
    let releaseFirst: () => void = () => undefined;
    pool.schedule("origin", new AbortController().signal, (release) => {
      releaseFirst = release;
    });
    const waiting = new AbortController();
    const skipped = vi.fn();
    pool.schedule("origin", waiting.signal, skipped);
    waiting.abort();
    releaseFirst();
    expect(skipped).not.toHaveBeenCalled();
    expect(pool.active("origin")).toBe(0);
  });
});
