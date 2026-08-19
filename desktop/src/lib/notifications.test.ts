import { describe, expect, it } from "vitest";
import {
  applyHostConnectivity,
  badgeCount,
  newlyCompletedJobs,
  shouldToastGenerationComplete,
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

describe("badgeCount", () => {
  it("hides at zero, shows a number, and caps at 99+", () => {
    expect(badgeCount(0)).toBeUndefined();
    expect(badgeCount(-1)).toBeUndefined();
    expect(badgeCount(3)).toBe(3);
    expect(badgeCount(150)).toBe("99+");
  });
});

describe("applyHostConnectivity", () => {
  function harness() {
    const warned = new Map<string, number>();
    const events: string[] = [];
    let nextToastId = 1;
    const live = new Set<number>();
    const effects = {
      warn: (host: HostStatusSnapshot) => {
        const id = nextToastId++;
        live.add(id);
        events.push(`warn:${host.id}:${id}`);
        return id;
      },
      announceRecovery: (host: HostStatusSnapshot) => void events.push(`recovered:${host.id}`),
      dismiss: (toastId: number) => {
        events.push(`dismiss:${toastId}${live.has(toastId) ? "" : ":gone"}`);
        live.delete(toastId);
      },
    };
    return { warned, events, effects, live };
  }

  const hosts = (...rows: [string, string][]) =>
    rows.map(([id, status]) => ({ id, label: id, status }));

  it("warns once on a drop and withdraws it on the recovery", () => {
    const h = harness();
    let snapshot = applyHostConnectivity({}, hosts(["a", "ready"]), h.warned, h.effects);
    snapshot = applyHostConnectivity(snapshot, hosts(["a", "error"]), h.warned, h.effects);
    snapshot = applyHostConnectivity(snapshot, hosts(["a", "error"]), h.warned, h.effects);
    expect(h.events).toEqual(["warn:a:1"]);

    snapshot = applyHostConnectivity(snapshot, hosts(["a", "ready"]), h.warned, h.effects);
    expect(h.events).toEqual(["warn:a:1", "dismiss:1", "recovered:a"]);
    expect(h.warned.size).toBe(0);
  });

  it("never celebrates a recovery it never warned about", () => {
    // A machine asleep at launch is errored but unannounced (the boot probe is
    // deliberately quiet); it waking up is not news.
    const h = harness();
    const snapshot = applyHostConnectivity({}, hosts(["a", "error"]), h.warned, h.effects);
    applyHostConnectivity(snapshot, hosts(["a", "ready"]), h.warned, h.effects);
    expect(h.events).toEqual([]);
  });

  it("retires a warning whose host leaves the list", () => {
    const h = harness();
    let snapshot = applyHostConnectivity({}, hosts(["a", "ready"]), h.warned, h.effects);
    snapshot = applyHostConnectivity(snapshot, hosts(["a", "error"]), h.warned, h.effects);
    snapshot = applyHostConnectivity(snapshot, [], h.warned, h.effects);
    expect(h.events).toEqual(["warn:a:1", "dismiss:1"]);
    expect(h.warned.size).toBe(0);
    // The host coming back later must not resurrect anything.
    applyHostConnectivity(snapshot, hosts(["a", "ready"]), h.warned, h.effects);
    expect(h.events).toEqual(["warn:a:1", "dismiss:1"]);
  });

  it("tolerates a user-dismissed warning at recovery time", () => {
    const h = harness();
    let snapshot = applyHostConnectivity({}, hosts(["a", "ready"]), h.warned, h.effects);
    snapshot = applyHostConnectivity(snapshot, hosts(["a", "error"]), h.warned, h.effects);
    h.live.delete(1); // the user closed it by hand
    applyHostConnectivity(snapshot, hosts(["a", "ready"]), h.warned, h.effects);
    expect(h.events).toEqual(["warn:a:1", "dismiss:1:gone", "recovered:a"]);
  });

  it("keeps each host's warning separate", () => {
    const h = harness();
    let snapshot = applyHostConnectivity(
      {},
      hosts(["a", "ready"], ["b", "ready"]),
      h.warned,
      h.effects,
    );
    snapshot = applyHostConnectivity(
      snapshot,
      hosts(["a", "error"], ["b", "error"]),
      h.warned,
      h.effects,
    );
    snapshot = applyHostConnectivity(
      snapshot,
      hosts(["a", "ready"], ["b", "error"]),
      h.warned,
      h.effects,
    );
    expect(h.events).toEqual(["warn:a:1", "warn:b:2", "dismiss:1", "recovered:a"]);
    expect([...h.warned.keys()]).toEqual(["b"]);
  });
});
