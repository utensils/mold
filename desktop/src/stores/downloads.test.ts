import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import {
  applyDownloadEvent,
  computeEtaSeconds,
  emptyDownloadsState,
  pushRateSample,
  useDownloadsStore,
  type RateSample,
} from "./downloads";
import { ApiError } from "../lib/api/client";
import type { DownloadEvent, DownloadJob, DownloadsListing } from "../lib/api/types";

const { startCatalogDownload } = vi.hoisted(() => ({
  startCatalogDownload: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("../lib/api/catalog", () => ({ startCatalogDownload }));

function reduce(events: DownloadEvent[]) {
  return events.reduce(applyDownloadEvent, emptyDownloadsState());
}

const snapshot: DownloadsListing = {
  active: {
    id: "a",
    model: "flux-dev:q8",
    status: "active",
    files_done: 1,
    files_total: 3,
    bytes_done: 1_000_000,
    bytes_total: 12_000_000,
  },
  queued: [
    {
      id: "b",
      model: "sdxl:fp16",
      status: "queued",
      files_done: 0,
      files_total: 0,
      bytes_done: 0,
      bytes_total: 0,
    },
  ],
  history: [],
};

describe("applyDownloadEvent", () => {
  it("seeds state from the snapshot frame", () => {
    const s = reduce([{ type: "snapshot", listing: snapshot }]);
    expect(s.activeJobs.map((job) => job.id)).toEqual(["a"]);
    expect(s.queued.map((j) => j.id)).toEqual(["b"]);
  });

  it("applies progress frames to the active job only", () => {
    const s = reduce([
      { type: "snapshot", listing: snapshot },
      {
        type: "progress",
        id: "a",
        files_done: 2,
        bytes_done: 8_000_000,
        current_file: "unet.safetensors",
      },
      { type: "progress", id: "b", files_done: 9, bytes_done: 9 }, // ignored — b isn't active
    ]);
    expect(s.activeJobs[0]?.bytes_done).toBe(8_000_000);
    expect(s.activeJobs[0]?.files_done).toBe(2);
    expect(s.activeJobs[0]?.current_file).toBe("unet.safetensors");
  });

  it("advances completed-file progress before the next byte update", () => {
    const s = reduce([
      { type: "snapshot", listing: snapshot },
      { type: "file_done", id: "a", filename: "tokenizer.json" },
    ]);
    expect(s.activeJobs[0]?.files_done).toBe(2);
  });

  it("promotes a queued job to active on started with its totals", () => {
    const s = reduce([
      { type: "snapshot", listing: { active: null, queued: snapshot.queued, history: [] } },
      { type: "started", id: "b", files_total: 4, bytes_total: 6_900_000 },
    ]);
    expect(s.activeJobs[0]?.id).toBe("b");
    expect(s.activeJobs[0]?.status).toBe("active");
    expect(s.activeJobs[0]?.bytes_total).toBe(6_900_000);
    expect(s.queued).toHaveLength(0);
  });

  it("keeps identity while early preparing totals are replaced by transfer totals", () => {
    const s = reduce([
      { type: "enqueued", id: "b", model: "ltx-2-19b-distilled:fp8", position: 0 },
      { type: "started", id: "b", files_total: 0, bytes_total: 0 },
      {
        type: "progress",
        id: "b",
        files_done: 0,
        bytes_done: 0,
        current_file: "Verifying file [1/19] tokenizer.json...",
      },
      { type: "started", id: "b", files_total: 19, bytes_total: 28_336_447_724 },
      { type: "enqueued", id: "c", model: "z-image-turbo:bf16", position: 0 },
      { type: "started", id: "c", files_total: 4, bytes_total: 12_000 },
      { type: "started", id: "b", files_total: 19, bytes_total: 28_336_447_724 },
    ]);
    expect(s.activeJobs.map((job) => job.id)).toEqual(["b", "c"]);
    expect(s.activeJobs[0]).toMatchObject({
      id: "b",
      model: "ltx-2-19b-distilled:fp8",
      status: "active",
      files_total: 19,
      bytes_total: 28_336_447_724,
    });
  });

  it("enqueues a synthetic queued job and dequeues by id", () => {
    const s = reduce([
      { type: "enqueued", id: "c", model: "z-image:q8", position: 1 },
      { type: "enqueued", id: "c", model: "z-image:q8", position: 1 }, // idempotent
    ]);
    expect(s.queued.map((j) => j.id)).toEqual(["c"]);
    const s2 = applyDownloadEvent(s, { type: "dequeued", id: "c" });
    expect(s2.queued).toHaveLength(0);
  });

  it("moves a completed job into history and clears active", () => {
    const s = reduce([
      { type: "snapshot", listing: snapshot },
      { type: "job_done", id: "a", model: "flux-dev:q8" },
    ]);
    expect(s.activeJobs).toHaveLength(0);
    expect(s.history[0]!.id).toBe("a");
    expect(s.history[0]!.status).toBe("completed");
  });

  it("records the error on a failed job", () => {
    const s = reduce([
      { type: "snapshot", listing: snapshot },
      { type: "job_failed", id: "a", error: "connection reset" },
    ]);
    expect(s.history[0]!.status).toBe("failed");
    expect(s.history[0]!.error).toBe("connection reset");
  });

  it("clears a model's stale failure when its retry completes", () => {
    const s = reduce([
      { type: "enqueued", id: "first", model: "minimax-h3:fp8", position: 0 },
      { type: "started", id: "first", files_total: 6, bytes_total: 100 },
      { type: "job_failed", id: "first", error: "permission denied" },
      { type: "enqueued", id: "retry", model: "minimax-h3:fp8", position: 0 },
      { type: "started", id: "retry", files_total: 6, bytes_total: 100 },
      { type: "job_done", id: "retry", model: "minimax-h3:fp8" },
    ]);

    expect(s.history).toHaveLength(1);
    expect(s.history[0]).toMatchObject({ id: "retry", status: "completed" });
    expect(s.history[0]!.error).toBeNull();
  });

  it("keeps failures for other models when a retry completes", () => {
    const s = reduce([
      { type: "enqueued", id: "other", model: "z-image:q6", position: 0 },
      { type: "job_failed", id: "other", error: "disk full" },
      { type: "enqueued", id: "retry", model: "minimax-h3:fp8", position: 0 },
      { type: "job_failed", id: "retry", error: "permission denied" },
      { type: "enqueued", id: "success", model: "minimax-h3:fp8", position: 0 },
      { type: "job_done", id: "success", model: "minimax-h3:fp8" },
    ]);

    expect(s.history.map((job) => [job.model, job.status])).toEqual([
      ["minimax-h3:fp8", "completed"],
      ["z-image:q6", "failed"],
    ]);
  });

  it("cancels a queued job into history", () => {
    const s = reduce([
      { type: "snapshot", listing: snapshot },
      { type: "job_cancelled", id: "b" },
    ]);
    expect(s.queued).toHaveLength(0);
    expect(s.history[0]!.id).toBe("b");
    expect(s.history[0]!.status).toBe("cancelled");
  });

  it("tracks progress for multiple active downloads independently", () => {
    const s = reduce([
      { type: "started", id: "a", files_total: 2, bytes_total: 100 },
      { type: "started", id: "b", files_total: 4, bytes_total: 200 },
      { type: "progress", id: "b", files_done: 1, bytes_done: 50 },
    ]);
    expect(s.activeJobs.map((job) => job.id)).toEqual(["a", "b"]);
    expect(s.activeJobs.find((job) => job.id === "a")?.bytes_done).toBe(0);
    expect(s.activeJobs.find((job) => job.id === "b")?.bytes_done).toBe(50);
  });
});

describe("computeEtaSeconds", () => {
  it("derives ETA from a steady byte rate", () => {
    const samples: RateSample[] = [
      { ts: 0, bytes: 0 },
      { ts: 5_000, bytes: 5_000_000 },
    ];
    // 1 MB/s with 15 MB remaining of 20 MB total → 15 s.
    expect(computeEtaSeconds(samples, 20_000_000)).toBe(15);
  });

  it("needs at least two samples", () => {
    expect(computeEtaSeconds([], 1_000)).toBeNull();
    expect(computeEtaSeconds([{ ts: 0, bytes: 10 }], 1_000)).toBeNull();
  });

  it("returns null for a stalled or non-advancing window", () => {
    const stalled: RateSample[] = [
      { ts: 0, bytes: 500 },
      { ts: 5_000, bytes: 500 },
    ];
    expect(computeEtaSeconds(stalled, 1_000)).toBeNull();
    const sameInstant: RateSample[] = [
      { ts: 1_000, bytes: 0 },
      { ts: 1_000, bytes: 100 },
    ];
    expect(computeEtaSeconds(sameInstant, 1_000)).toBeNull();
  });
});

describe("pushRateSample", () => {
  it("trims samples that fall out of the sliding window", () => {
    const samples: RateSample[] = [];
    pushRateSample(samples, 0, 0);
    pushRateSample(samples, 100, 5_000);
    pushRateSample(samples, 200, 12_000); // first sample is now >10 s old
    expect(samples.map((s) => s.ts)).toEqual([5_000, 12_000]);
  });
});

function failedJob(overrides: Partial<DownloadJob> = {}): DownloadJob {
  return {
    id: "f1",
    model: "flux-dev:q4",
    status: "failed",
    files_done: 1,
    files_total: 4,
    bytes_done: 25,
    bytes_total: 100,
    error: "connection reset",
    ...overrides,
  };
}

describe("downloads store ETA + history + retry", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    startCatalogDownload.mockReset();
    startCatalogDownload.mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("exposes a live ETA per active job from progress samples", () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const store = useDownloadsStore();
    store.apply({ type: "started", id: "a", files_total: 2, bytes_total: 100_000_000 });
    store.apply({ type: "progress", id: "a", files_done: 0, bytes_done: 0 });
    vi.setSystemTime(10_000);
    store.apply({ type: "progress", id: "a", files_done: 1, bytes_done: 10_000_000 });
    // 1 MB/s with 90 MB remaining → 90 s.
    expect(store.etaByJob["a"]).toBe(90);
  });

  it("exposes the live transfer rate per active job from the same samples", () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const store = useDownloadsStore();
    store.apply({ type: "started", id: "a", files_total: 2, bytes_total: 100_000_000 });
    store.apply({ type: "progress", id: "a", files_done: 0, bytes_done: 0 });
    // One sample is no rate yet — the banner shows a dash, never a guess.
    expect(store.rateByJob["a"]).toBeNull();
    vi.setSystemTime(10_000);
    store.apply({ type: "progress", id: "a", files_done: 1, bytes_done: 10_000_000 });
    expect(store.rateByJob["a"]).toBe(1_000_000);
  });

  it("drops the rate window when a job settles", () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const store = useDownloadsStore();
    store.apply({ type: "started", id: "a", files_total: 1, bytes_total: 100 });
    store.apply({ type: "progress", id: "a", files_done: 0, bytes_done: 10 });
    vi.setSystemTime(1_000);
    store.apply({ type: "progress", id: "a", files_done: 0, bytes_done: 20 });
    store.apply({ type: "job_failed", id: "a", error: "boom" });
    expect(store.etaByJob["a"]).toBeUndefined();
    expect(store.rateByJob["a"]).toBeUndefined();
    expect(Object.keys(store.rateSamples)).toHaveLength(0);
  });

  it("surfaces failed jobs from every host in hostedHistory", () => {
    const store = useDownloadsStore();
    store.apply({ type: "snapshot", listing: { active: null, queued: [], history: [] } });
    store.activeJobs = [
      { ...failedJob({ id: "p1", model: "sdxl:fp16" }), status: "active" as const },
    ];
    store.apply({ type: "job_failed", id: "p1", error: "disk full" });
    store.hostStates["hal9000"] = {
      label: "hal9000",
      target: { baseUrl: "http://hal9000:7680", apiKey: "remote-key" },
      activeJobs: [],
      queued: [],
      history: [failedJob()],
      subscribed: true,
      abort: null,
      cancelling: [],
      ready: null,
    };
    const rows = store.hostedHistory;
    expect(rows.map((r) => `${r.hostId}:${r.job.id}`)).toEqual(["primary:p1", "hal9000:f1"]);
    expect(rows[0]!.job.status).toBe("failed");
    expect(rows[0]!.job.error).toBe("disk full");
  });

  it("orders each host's history newest-first by completion time", () => {
    const store = useDownloadsStore();
    store.hostStates["hal9000"] = {
      label: "hal9000",
      target: { baseUrl: "http://hal9000:7680", apiKey: null },
      activeJobs: [],
      queued: [],
      history: [
        failedJob({ id: "old", completed_at: 1_000 }),
        failedJob({ id: "new", completed_at: 3_000 }),
        failedJob({ id: "mid", completed_at: 2_000 }),
      ],
      subscribed: true,
      abort: null,
      cancelling: [],
      ready: null,
    };
    expect(store.hostedHistory.map((r) => r.job.id)).toEqual(["new", "mid", "old"]);
  });

  it("keeps the server's relative order for history entries without a timestamp", () => {
    const store = useDownloadsStore();
    store.history = [
      failedJob({ id: "first" }),
      failedJob({ id: "second" }),
      failedJob({ id: "third" }),
    ];
    expect(store.hostedHistory.map((r) => r.job.id)).toEqual(["first", "second", "third"]);
  });

  it("retries a failed job on the same extra host, forwarding credentials", async () => {
    const store = useDownloadsStore();
    store.hostStates["hal9000"] = {
      label: "hal9000",
      target: { baseUrl: "http://hal9000:7680", apiKey: "remote-key" },
      activeJobs: [],
      queued: [],
      history: [failedJob()],
      subscribed: true,
      abort: null,
      cancelling: [],
      ready: null,
    };
    await store.retry("hal9000", failedJob());
    expect(startCatalogDownload).toHaveBeenCalledWith(
      "flux-dev:q4",
      { baseUrl: "http://hal9000:7680", apiKey: "remote-key" },
      true,
    );
  });

  it("retries a failed primary job against the primary target without forwarding", async () => {
    const store = useDownloadsStore();
    store.primaryHostId = "local";
    store.primaryTarget = { baseUrl: "http://127.0.0.1:49152", apiKey: "local-key" };
    await store.retry("local", failedJob({ model: "cv:8001" }));
    expect(startCatalogDownload).toHaveBeenCalledWith(
      "cv:8001",
      { baseUrl: "http://127.0.0.1:49152", apiKey: "local-key" },
      false,
    );
  });

  it("swallows an already-queued conflict on retry", async () => {
    startCatalogDownload.mockRejectedValueOnce(new ApiError("conflict", 409));
    const store = useDownloadsStore();
    store.primaryHostId = "local";
    store.primaryTarget = { baseUrl: "http://127.0.0.1:49152", apiKey: null };
    await expect(store.retry("local", failedJob())).resolves.toBeUndefined();
  });
});
