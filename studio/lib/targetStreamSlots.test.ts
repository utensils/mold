import { describe, expect, it, vi } from "vitest";
import { TargetStreamSlots } from "./targetStreamSlots";

describe("TargetStreamSlots", () => {
  it("drains an arbitrary backlog independently per target", () => {
    const slots = new TargetStreamSlots(4);
    const started: string[] = [];
    const releases = new Map<string, () => void>();

    for (const target of ["alpha", "beta"]) {
      for (let index = 0; index < 257; index += 1) {
        const id = `${target}-${index}`;
        slots.schedule(target, new AbortController().signal, (release) => {
          started.push(id);
          releases.set(id, release);
        });
      }
    }

    expect(started.filter((id) => id.startsWith("alpha-"))).toEqual([
      "alpha-0",
      "alpha-1",
      "alpha-2",
      "alpha-3",
    ]);
    expect(started.filter((id) => id.startsWith("beta-"))).toEqual([
      "beta-0",
      "beta-1",
      "beta-2",
      "beta-3",
    ]);

    releases.get("alpha-0")!();
    expect(started.at(-1)).toBe("alpha-4");
    expect(slots.active("alpha")).toBe(4);
    expect(slots.waiting("alpha")).toBe(252);
    expect(slots.active("beta")).toBe(4);
    expect(slots.waiting("beta")).toBe(253);
  });

  it("removes a cancelled waiter without starting or consuming a slot", () => {
    const slots = new TargetStreamSlots(1);
    let releaseActive: () => void = () => undefined;
    slots.schedule("render-host", new AbortController().signal, (release) => {
      releaseActive = release;
    });
    const waiting = new AbortController();
    const startWaiting = vi.fn();
    slots.schedule("render-host", waiting.signal, startWaiting);

    waiting.abort();
    releaseActive();

    expect(startWaiting).not.toHaveBeenCalled();
    expect(slots.active("render-host")).toBe(0);
    expect(slots.waiting("render-host")).toBe(0);
  });

  it("supports the awaitable acquisition used by desktop and mobile", async () => {
    const slots = new TargetStreamSlots(1);
    const first = new AbortController();
    const releaseFirst = await slots.acquire("phone-host", first.signal);
    const second = new AbortController();
    const waiting = slots.acquire("phone-host", second.signal);

    expect(slots.active("phone-host")).toBe(1);
    expect(slots.waiting("phone-host")).toBe(1);
    releaseFirst!();
    const releaseSecond = await waiting;
    expect(releaseSecond).toBeTypeOf("function");
    releaseSecond!();
    expect(slots.active("phone-host")).toBe(0);
  });
});
