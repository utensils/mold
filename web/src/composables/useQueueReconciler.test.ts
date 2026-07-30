import { afterEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import {
  reconcileRound,
  startGenerateQueueReconciler,
  startQueueReconciler,
  RECONCILE_GRACE_MS,
} from "./useQueueReconciler";
import type { Job } from "./useGenerateStream";
import type { GenerateRequestWire } from "../types";

vi.mock("../api", () => ({
  fetchQueue: vi.fn(),
}));

import { fetchQueue } from "../api";

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "client-1",
    request: {
      prompt: "x",
      model: "flux-dev:fp16",
      width: 512,
      height: 512,
      steps: 8,
      guidance: 3,
      strength: 1,
      fps: 24,
      output_format: "png",
      frames: 1,
    } as GenerateRequestWire,
    startedAt: 0,
    controller: new AbortController(),
    progress: {
      stage: "Denoising",
      step: 1,
      totalSteps: 10,
      weightBytesLoaded: null,
      weightBytesTotal: null,
      queuePosition: null,
      gpu: null,
      elapsedMs: 0,
    },
    result: null,
    error: null,
    state: "running",
    settledAt: null,
    chain: null,
    lastProgressAt: 1000,
    workStarted: true,
    hostId: null,
    hostLabel: null,
    target: null,
    serverId: "srv-1",
    previewUrl: null,
    seedVisual: "flux-dev:fp16·x",
    ...overrides,
  };
}

afterEach(() => {
  vi.mocked(fetchQueue).mockReset();
});

describe("reconcileRound (pure)", () => {
  it("reports a running job whose serverId is absent server-side past the grace window", () => {
    const job = makeJob({ lastProgressAt: 1000 });
    const jobs = [job];
    // No serverIds in the listing — the job is a zombie.
    expect(
      reconcileRound(jobs, new Set(), 1000 + RECONCILE_GRACE_MS + 1),
    ).toEqual([job]);
    expect(job.state).toBe("running");
    expect(job.error).toBeNull();
  });

  it("leaves a running job alone when the server still knows about it", () => {
    const job = makeJob({ serverId: "srv-keep" });
    const jobs = [job];
    expect(
      reconcileRound(
        jobs,
        new Set(["srv-keep"]),
        job.lastProgressAt + RECONCILE_GRACE_MS + 1,
      ),
    ).toEqual([]);
    expect(job.state).toBe("running");
    expect(job.error).toBeNull();
  });

  it("does NOT dead-letter inside the grace window (covers submit→first-event race)", () => {
    // Card just opened — no progress event has landed yet. The server may
    // have the job, but a poll racing the SSE handshake would see an empty
    // serverId set. We must NOT flip the card.
    const job = makeJob({ lastProgressAt: 1000 });
    expect(reconcileRound([job], new Set(), 1000 + 1000)).toEqual([]); // 1 s after start
    expect(job.state).toBe("running");
  });

  it("skips jobs before the queued-event handshake captures a serverId", () => {
    // Without a serverId the reconciler can't tell server-side reality
    // from "we never had an id." Skipping is the safe default — the L2
    // staleness badge and the L1 silent-close fix cover those cases.
    const job = makeJob({ serverId: null, lastProgressAt: 0 });
    expect(reconcileRound([job], new Set(), 1_000_000)).toEqual([]);
    expect(job.state).toBe("running");
  });

  it("skips jobs that aren't in `running` state", () => {
    // Errored / done / canceled cards aren't candidates for the sweep —
    // running is the only state that gets polled.
    const done = makeJob({ state: "done", serverId: "srv-done" });
    const errored = makeJob({ state: "error", serverId: "srv-err" });
    expect(
      reconcileRound(
        [done, errored],
        new Set(),
        done.lastProgressAt + RECONCILE_GRACE_MS + 1,
      ),
    ).toEqual([]);
    expect(done.state).toBe("done");
    expect(errored.state).toBe("error");
  });
});

describe("startQueueReconciler (live polling)", () => {
  it("wires a generation stream's failure owner into reconciliation", async () => {
    vi.useFakeTimers();
    vi.mocked(fetchQueue).mockResolvedValue({ entries: [] });
    const jobs = ref<Job[]>([
      makeJob({ id: "missing-client", lastProgressAt: 0 }),
    ]);
    const failRunning = vi.fn();
    const handle = startGenerateQueueReconciler(
      { jobs, failRunning },
      { intervalMs: 1_000 },
    );

    await vi.advanceTimersByTimeAsync(RECONCILE_GRACE_MS + 2_100);

    expect(failRunning).toHaveBeenCalledWith(
      "missing-client",
      "job not found on server — connection lost",
    );
    handle.stop();
    vi.useRealTimers();
  });

  it("calls fetchQueue and reconciles when running jobs have serverIds", async () => {
    vi.useFakeTimers();
    vi.mocked(fetchQueue).mockResolvedValue({
      entries: [
        {
          id: "srv-keep",
          model: "flux-dev:fp16",
          state: "running",
          started_at_unix_ms: 0,
          position: 0,
        },
      ],
    });

    const jobs = ref<Job[]>([
      makeJob({ id: "j-keep", serverId: "srv-keep", lastProgressAt: 0 }),
      makeJob({ id: "j-zombie", serverId: "srv-zombie", lastProgressAt: 0 }),
    ]);
    const failed: Array<{ id: string; error: string }> = [];
    const handle = startQueueReconciler(
      jobs,
      (id, error) => {
        failed.push({ id, error });
        const job = jobs.value.find((candidate) => candidate.id === id);
        if (job?.state === "running") job.state = "error";
      },
      { intervalMs: 1_000 },
    );

    // Advance past the first scheduled tick.
    await vi.advanceTimersByTimeAsync(1_001);
    // Then past the grace window so the missing job qualifies for
    // dead-letter on the next round.
    await vi.advanceTimersByTimeAsync(RECONCILE_GRACE_MS + 1_001);

    expect(fetchQueue).toHaveBeenCalled();
    expect(jobs.value[0].state).toBe("running");
    expect(jobs.value[1].state).toBe("error");
    expect(failed).toEqual([
      {
        id: "j-zombie",
        error: "job not found on server — connection lost",
      },
    ]);

    handle.stop();
    vi.useRealTimers();
  });

  it("polls each job's routed host and reconciles only against that host", async () => {
    vi.useFakeTimers();
    const studio = { baseUrl: "http://studio:7680", apiKey: "sk-studio" };
    vi.mocked(fetchQueue).mockImplementation(async (target) => ({
      entries:
        target?.baseUrl === studio.baseUrl
          ? [
              {
                id: "remote-keep",
                model: "flux-dev:fp16",
                state: "running",
                started_at_unix_ms: 0,
                position: 0,
              },
            ]
          : [],
    }));
    const remote = makeJob({
      id: "remote-client",
      serverId: "remote-keep",
      hostId: "studio",
      target: studio,
      lastProgressAt: 0,
    });
    const originZombie = makeJob({
      id: "origin-client",
      serverId: "origin-gone",
      lastProgressAt: 0,
    });
    const jobs = ref([remote, originZombie]);
    const handle = startQueueReconciler(
      jobs,
      (id, error) => {
        const job = jobs.value.find((candidate) => candidate.id === id);
        if (job?.state === "running") {
          job.state = "error";
          job.error = error;
        }
      },
      { intervalMs: 1_000 },
    );

    await vi.advanceTimersByTimeAsync(RECONCILE_GRACE_MS + 2_100);

    expect(fetchQueue).toHaveBeenCalledWith(studio);
    expect(fetchQueue).toHaveBeenCalledWith(undefined);
    expect(remote.state).toBe("running");
    expect(originZombie.state).toBe("error");
    handle.stop();
    vi.useRealTimers();
  });

  it("does NOT poll the server when there are no running candidates", async () => {
    vi.useFakeTimers();
    const jobs = ref<Job[]>([makeJob({ state: "done" })]);
    const handle = startQueueReconciler(jobs, () => {}, { intervalMs: 500 });

    await vi.advanceTimersByTimeAsync(2_000);
    expect(fetchQueue).not.toHaveBeenCalled();

    handle.stop();
    vi.useRealTimers();
  });

  it("stop() prevents any further fetchQueue calls", async () => {
    vi.useFakeTimers();
    vi.mocked(fetchQueue).mockResolvedValue({ entries: [] });
    const jobs = ref<Job[]>([makeJob()]);
    const handle = startQueueReconciler(jobs, () => {}, { intervalMs: 500 });

    handle.stop();
    await vi.advanceTimersByTimeAsync(5_000);
    expect(fetchQueue).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("does not dead-letter when the server itself is unreachable (transient)", async () => {
    // A poll failure could mean either "server down" or "transient blip."
    // Either way the SSE per-job error path is the right place to mark
    // a card as dead — the reconciler MUST NOT take that action on its
    // own from a poll failure.
    vi.useFakeTimers();
    vi.mocked(fetchQueue).mockRejectedValue(new Error("ECONNREFUSED"));

    const job = makeJob({ lastProgressAt: 0 });
    const jobs = ref<Job[]>([job]);
    const handle = startQueueReconciler(jobs, () => {}, { intervalMs: 500 });

    await vi.advanceTimersByTimeAsync(600);
    // Flush the rejected fetch microtask.
    await Promise.resolve();
    await Promise.resolve();
    expect(job.state).toBe("running");

    handle.stop();
    vi.useRealTimers();
  });
});
