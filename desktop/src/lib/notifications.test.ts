import { describe, expect, it } from "vitest";
import {
  badgeCount,
  detectOfflineTransitions,
  newlyCompletedJobs,
  shouldToastGenerationComplete,
  snapshotHostStatuses,
  type HostStatusSnapshot,
} from "./notifications";
import type { Job } from "./generationJob";

const job = (clientId: number, status: string): Job => ({ clientId, status }) as unknown as Job;

describe("newlyCompletedJobs", () => {
  it("returns completed jobs not yet seen", () => {
    const jobs = [job(1, "complete"), job(2, "denoising"), job(3, "complete"), job(4, "error")];
    const seen = new Set<number>();
    const done = newlyCompletedJobs(jobs, seen);
    expect(done.map((j) => j.clientId)).toEqual([1, 3]);
  });

  it("does not re-report a completion already in the seen set", () => {
    const jobs = [job(1, "complete")];
    expect(newlyCompletedJobs(jobs, new Set([1]))).toHaveLength(0);
  });
});

describe("shouldToastGenerationComplete", () => {
  it("suppresses the toast on the Create canvas but shows it elsewhere", () => {
    expect(shouldToastGenerationComplete("/create")).toBe(false);
    expect(shouldToastGenerationComplete("/library")).toBe(true);
    expect(shouldToastGenerationComplete("/machines")).toBe(true);
  });
});

describe("detectOfflineTransitions", () => {
  const hosts = (...rows: Array<[string, string]>): HostStatusSnapshot[] =>
    rows.map(([id, status]) => ({ id, label: id, status }));

  it("fires only on the ready → error edge", () => {
    const prev = snapshotHostStatuses(hosts(["a", "ready"], ["b", "ready"]));
    const offline = detectOfflineTransitions(prev, hosts(["a", "error"], ["b", "ready"]));
    expect(offline.map((h) => h.id)).toEqual(["a"]);
  });

  it("does not re-fire while the host stays offline", () => {
    const prev = snapshotHostStatuses(hosts(["a", "error"]));
    expect(detectOfflineTransitions(prev, hosts(["a", "error"]))).toHaveLength(0);
  });

  it("ignores a host that boots straight into error (not a transition)", () => {
    expect(detectOfflineTransitions({}, hosts(["a", "error"]))).toHaveLength(0);
  });

  it("ignores connecting → error (a reconnect attempt, not a live host dropping)", () => {
    const prev = snapshotHostStatuses(hosts(["a", "connecting"]));
    expect(detectOfflineTransitions(prev, hosts(["a", "error"]))).toHaveLength(0);
  });
});

describe("badgeCount", () => {
  it("hides at zero, shows a number, and caps at 99+", () => {
    expect(badgeCount(0)).toBeUndefined();
    expect(badgeCount(-1)).toBeUndefined();
    expect(badgeCount(3)).toBe(3);
    expect(badgeCount(150)).toBe("99+");
  });
});
