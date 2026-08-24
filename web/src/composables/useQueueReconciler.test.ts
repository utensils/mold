import { afterEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import {
  reconcileRound,
  startGenerateQueueReconciler,
  startQueueReconciler,
  targetForJob,
  DETACHED_SETTLE_NOTE,
  RECONCILE_GRACE_MS,
} from "./useQueueReconciler";
import type { Job } from "./useGenerateStream";
import type { GenerateRequestWire } from "../types";

vi.mock("../api", () => ({
  fetchQueue: vi.fn(),
  listGalleryFrom: vi.fn().mockResolvedValue([]),
}));

import { fetchQueue, listGalleryFrom } from "../api";

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
      // Never started work: no output can exist, so a vanished row is a
      // genuine dead letter rather than a print waiting in the Library.
      makeJob({ id: "missing-client", lastProgressAt: 0, workStarted: false }),
    ]);
    const failRunning = vi.fn();
    const handle = startGenerateQueueReconciler(
      { jobs, failRunning, settleDetached: vi.fn() },
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

  it("does not dead-letter a job whose row vanished because it COMPLETED", async () => {
    // Evidence, not inference: the host's gallery carries a print stamped with
    // this job's id, so the row left the queue by finishing.
    vi.useFakeTimers();
    vi.mocked(fetchQueue).mockResolvedValue({ entries: [] });
    vi.mocked(listGalleryFrom).mockResolvedValue([
      {
        filename: "finished.png",
        timestamp: 0,
        metadata: { job_id: "srv-1" },
      },
    ] as never);
    const jobs = ref<Job[]>([
      makeJob({ id: "finished-client", lastProgressAt: 0, workStarted: true }),
    ]);
    const failRunning = vi.fn();
    const settleDetached = vi.fn();
    const handle = startGenerateQueueReconciler(
      { jobs, failRunning, settleDetached },
      { intervalMs: 1_000 },
    );

    await vi.advanceTimersByTimeAsync(RECONCILE_GRACE_MS + 2_100);

    expect(failRunning).not.toHaveBeenCalled();
    expect(settleDetached).toHaveBeenCalledWith(
      "finished-client",
      DETACHED_SETTLE_NOTE,
    );
    handle.stop();
    vi.useRealTimers();
  });

  it("keeps the failure path for work that started and then genuinely died", async () => {
    // No print, and the host never promised to keep it. Soft-settling here
    // hides the row (the strip retires detached rows) AND shows no output —
    // silence, which is worse than the dead letter it replaced.
    vi.useFakeTimers();
    vi.mocked(fetchQueue).mockResolvedValue({ entries: [] });
    vi.mocked(listGalleryFrom).mockResolvedValue([] as never);
    const jobs = ref<Job[]>([
      makeJob({ id: "died-client", lastProgressAt: 0, workStarted: true }),
    ]);
    const failRunning = vi.fn();
    const settleDetached = vi.fn();
    const handle = startGenerateQueueReconciler(
      { jobs, failRunning, settleDetached },
      { intervalMs: 1_000 },
    );

    await vi.advanceTimersByTimeAsync(RECONCILE_GRACE_MS + 2_100);

    expect(settleDetached).not.toHaveBeenCalled();
    expect(failRunning).toHaveBeenCalledWith(
      "died-client",
      "job not found on server — connection lost",
    );
    handle.stop();
    vi.useRealTimers();
  });

  it("accepts the host's own promise as evidence when no print exists yet", async () => {
    // A durable host said it journalled this job at admission: it will run,
    // even though nothing has landed in the gallery yet.
    vi.useFakeTimers();
    vi.mocked(fetchQueue).mockResolvedValue({ entries: [] });
    vi.mocked(listGalleryFrom).mockResolvedValue([] as never);
    const jobs = ref<Job[]>([
      makeJob({
        id: "retained-client",
        lastProgressAt: 0,
        workStarted: true,
        durable: true,
      }),
    ]);
    const failRunning = vi.fn();
    const settleDetached = vi.fn();
    const handle = startGenerateQueueReconciler(
      { jobs, failRunning, settleDetached },
      { intervalMs: 1_000 },
    );

    await vi.advanceTimersByTimeAsync(RECONCILE_GRACE_MS + 2_100);

    expect(failRunning).not.toHaveBeenCalled();
    expect(settleDetached).toHaveBeenCalledWith(
      "retained-client",
      DETACHED_SETTLE_NOTE,
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
      makeJob({
        id: "j-zombie",
        serverId: "srv-zombie",
        lastProgressAt: 0,
        workStarted: false,
      }),
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

  it.each([
    [
      "live-only",
      {
        entries: [],
        live_only_entries: [
          {
            id: "srv-1",
            model: "m",
            state: "running",
            started_at_unix_ms: 0,
            position: 0,
          },
        ],
      },
    ],
    [
      "truncated",
      {
        entries: [],
        page: { limit: 1, offset: 0, returned: 0, next_cursor: "older" },
      },
    ],
  ])(
    "does not dead-letter a %s bounded queue result",
    async (_label, listing) => {
      vi.useFakeTimers();
      vi.mocked(fetchQueue).mockResolvedValue(listing as never);
      const jobs = ref<Job[]>([
        makeJob({ lastProgressAt: 0, workStarted: false }),
      ]);
      const failRunning = vi.fn();
      const handle = startQueueReconciler(jobs, failRunning, {
        intervalMs: 1_000,
      });

      await vi.advanceTimersByTimeAsync(RECONCILE_GRACE_MS + 2_100);

      expect(failRunning).not.toHaveBeenCalled();
      handle.stop();
      vi.useRealTimers();
    },
  );

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

  it("settles a detached job with the check-the-Library note, not a failure", async () => {
    vi.useFakeTimers();
    vi.mocked(fetchQueue).mockResolvedValue({ entries: [] } as never);

    const job = makeJob({ detached: true, lastProgressAt: 0 });
    const jobs = ref<Job[]>([job]);
    const failRunning = vi.fn();
    const settleDetached = vi.fn();
    const handle = startQueueReconciler(jobs, failRunning, {
      intervalMs: 500,
      settleDetached,
    });

    await vi.advanceTimersByTimeAsync(600);
    await Promise.resolve();
    await Promise.resolve();

    expect(failRunning).not.toHaveBeenCalled();
    expect(settleDetached).toHaveBeenCalledWith(
      "client-1",
      DETACHED_SETTLE_NOTE,
    );
    expect(DETACHED_SETTLE_NOTE).toContain("Library");
    expect(DETACHED_SETTLE_NOTE.toLowerCase()).not.toContain("failed");

    handle.stop();
    vi.useRealTimers();
  });

  it("resolves a reloaded job's route from the host registry", () => {
    localStorage.setItem(
      "mold.web.hosts.v1",
      JSON.stringify([
        {
          id: "studio-2-7680",
          name: "Studio 2",
          url: "http://studio-2:7680",
          apiKey: "secret",
          connected: true,
        },
      ]),
    );
    try {
      // In-memory target always wins (it carries the exact key used).
      const live = makeJob({
        target: { baseUrl: "http://live:7680" },
        hostId: "studio-2-7680",
      });
      expect(targetForJob(live)).toEqual({ baseUrl: "http://live:7680" });
      // A reloaded job (target died with the session) resolves via registry.
      const reloaded = makeJob({ target: null, hostId: "studio-2-7680" });
      expect(targetForJob(reloaded)).toEqual({
        baseUrl: "http://studio-2:7680",
        apiKey: "secret",
      });
      // Origin (or unknown) hosts reconcile against the primary connection.
      expect(targetForJob(makeJob({ target: null, hostId: null }))).toBeNull();
    } finally {
      localStorage.removeItem("mold.web.hosts.v1");
    }
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
