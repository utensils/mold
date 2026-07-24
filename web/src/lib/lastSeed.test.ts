import { beforeEach, describe, expect, it } from "vitest";
import {
  completedSeedToLock,
  LAST_SEED_STORAGE_KEY,
  loadLastSeed,
  storeLastSeed,
} from "./lastSeed";

describe("last-seed persistence", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("round-trips a stored seed", () => {
    storeLastSeed(184023);
    expect(loadLastSeed()).toBe(184023);
  });

  it("returns null when nothing is stored", () => {
    expect(loadLastSeed()).toBeNull();
  });

  it("returns null for malformed or non-finite stored values", () => {
    localStorage.setItem(LAST_SEED_STORAGE_KEY, "not-a-number");
    expect(loadLastSeed()).toBeNull();
    localStorage.setItem(LAST_SEED_STORAGE_KEY, "Infinity");
    expect(loadLastSeed()).toBeNull();
    localStorage.setItem(LAST_SEED_STORAGE_KEY, "-5");
    expect(loadLastSeed()).toBeNull();
  });

  it("survives a job-rail wipe — the key is independent of mold.generate.jobs", () => {
    storeLastSeed(7);
    localStorage.removeItem("mold.generate.jobs");
    expect(loadLastSeed()).toBe(7);
  });
});

describe("completedSeedToLock", () => {
  it("accepts a single-clip completion's server-reported seed", () => {
    expect(
      completedSeedToLock({
        chain: null,
        request: { seed: null },
        result: { seed_used: 42 },
      }),
    ).toBe(42);
  });

  it("rejects a random-seed chain completion (fabricated seed_used of 0)", () => {
    // chainCompleteToSingle falls back to `req.seed ?? 0` — a chain that ran
    // with a random seed reports 0, which is not a real seed to lock.
    expect(
      completedSeedToLock({
        chain: { stageCount: 3, currentStage: 2, estimatedTotalFrames: null },
        request: {},
        result: { seed_used: 0 },
      }),
    ).toBeNull();
  });

  it("accepts a chain completion whose request pinned an explicit seed", () => {
    expect(
      completedSeedToLock({
        chain: { stageCount: 3, currentStage: 2, estimatedTotalFrames: null },
        request: { seed: 99 },
        result: { seed_used: 99 },
      }),
    ).toBe(99);
  });

  it("rejects missing results and undefined/non-finite seeds", () => {
    expect(
      completedSeedToLock({ chain: null, request: {}, result: null }),
    ).toBeNull();
    expect(
      completedSeedToLock({
        chain: null,
        request: {},
        result: { seed_used: undefined as unknown as number },
      }),
    ).toBeNull();
    expect(
      completedSeedToLock({
        chain: null,
        request: {},
        result: { seed_used: Number.NaN },
      }),
    ).toBeNull();
  });
});
