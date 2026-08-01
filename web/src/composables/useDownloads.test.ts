import { describe, it, expect } from "vitest";
import type {
  DownloadJobWire,
  JobStatusWire,
  DownloadEventWire,
} from "../types";

describe("downloads types", () => {
  it("job status values match server enum", () => {
    const all: JobStatusWire[] = [
      "queued",
      "active",
      "completed",
      "failed",
      "cancelled",
    ];
    expect(all).toHaveLength(5);
  });

  it("download job shape", () => {
    const job: DownloadJobWire = {
      id: "abc",
      model: "flux-dev:q4",
      status: "active",
      files_done: 0,
      files_total: 5,
      bytes_done: 0,
      bytes_total: 10_000,
      current_file: null,
      started_at: null,
      completed_at: null,
      error: null,
    };
    expect(job.status).toBe("active");
  });

  it("event discriminator shapes", () => {
    const e: DownloadEventWire = {
      type: "enqueued",
      id: "x",
      model: "flux-dev:q4",
      position: 1,
    };
    expect(e.type).toBe("enqueued");
  });
});

import {
  applyDownloadEvent,
  computeEtaSeconds,
  newDownloadsState,
} from "./useDownloads";
import type { DownloadEventWire as DEW } from "../types";

describe("computeEtaSeconds", () => {
  it("returns null when fewer than 2 samples", () => {
    expect(computeEtaSeconds([], 1000)).toBeNull();
    expect(computeEtaSeconds([{ ts: 0, bytes: 0 }], 1000)).toBeNull();
  });

  it("computes eta from sliding window", () => {
    const history = [
      { ts: 0, bytes: 0 },
      { ts: 2000, bytes: 1_000_000 },
    ];
    // rate = 500_000 B/s; remaining = 1_000_000; eta = 2 s
    expect(computeEtaSeconds(history, 2_000_000)).toBe(2);
  });

  it("returns null when rate is non-positive", () => {
    const history = [
      { ts: 0, bytes: 1_000_000 },
      { ts: 1000, bytes: 1_000_000 },
    ];
    expect(computeEtaSeconds(history, 2_000_000)).toBeNull();
  });
});

describe("applyDownloadEvent", () => {
  it("Enqueued appends a queued job", () => {
    const state = newDownloadsState();
    const evt: DEW = {
      type: "enqueued",
      id: "a",
      model: "flux-dev:q4",
      position: 1,
    };
    applyDownloadEvent(state, evt);
    expect(state.queued).toHaveLength(1);
    expect(state.queued[0].id).toBe("a");
    expect(state.queued[0].status).toBe("queued");
  });

  it("Started promotes queued->active", () => {
    const state = newDownloadsState();
    applyDownloadEvent(state, {
      type: "enqueued",
      id: "a",
      model: "flux-dev:q4",
      position: 1,
    });
    applyDownloadEvent(state, {
      type: "started",
      id: "a",
      files_total: 3,
      bytes_total: 9_000,
    });
    expect(state.queued).toHaveLength(0);
    expect(state.active?.id).toBe("a");
    expect(state.active?.files_total).toBe(3);
    expect(state.active?.bytes_total).toBe(9_000);
    expect(state.active?.status).toBe("active");
  });

  it("preserves the model when preparing is followed by real transfer totals", () => {
    const state = newDownloadsState();
    applyDownloadEvent(state, {
      type: "enqueued",
      id: "a",
      model: "ltx-2-19b-distilled:fp8",
      position: 0,
    });
    applyDownloadEvent(state, {
      type: "started",
      id: "a",
      files_total: 0,
      bytes_total: 0,
    });
    applyDownloadEvent(state, {
      type: "started",
      id: "a",
      files_total: 19,
      bytes_total: 28_336_447_724,
    });
    expect(state.activeJobs).toHaveLength(1);
    expect(state.active).toMatchObject({
      id: "a",
      model: "ltx-2-19b-distilled:fp8",
      files_total: 19,
      bytes_total: 28_336_447_724,
    });
  });

  it("Progress updates bytes_done", () => {
    const state = newDownloadsState();
    applyDownloadEvent(state, {
      type: "enqueued",
      id: "a",
      model: "m",
      position: 1,
    });
    applyDownloadEvent(state, {
      type: "started",
      id: "a",
      files_total: 1,
      bytes_total: 1000,
    });
    applyDownloadEvent(state, {
      type: "progress",
      id: "a",
      files_done: 0,
      bytes_done: 500,
      current_file: "foo.bin",
    });
    expect(state.active?.bytes_done).toBe(500);
    expect(state.active?.current_file).toBe("foo.bin");
  });

  it("tracks two active downloads independently", () => {
    const state = newDownloadsState();
    for (const id of ["a", "b"]) {
      applyDownloadEvent(state, {
        type: "enqueued",
        id,
        model: id,
        position: 1,
      });
      applyDownloadEvent(state, {
        type: "started",
        id,
        files_total: 1,
        bytes_total: 100,
      });
    }
    applyDownloadEvent(state, {
      type: "progress",
      id: "b",
      files_done: 0,
      bytes_done: 50,
    });
    expect(state.activeJobs.map((job) => job.id)).toEqual(["a", "b"]);
    expect(state.activeJobs.find((job) => job.id === "a")?.bytes_done).toBe(0);
    expect(state.activeJobs.find((job) => job.id === "b")?.bytes_done).toBe(50);
  });

  it("JobDone moves active into history, capped at 20", () => {
    const state = newDownloadsState();
    for (let i = 0; i < 25; i++) {
      applyDownloadEvent(state, {
        type: "enqueued",
        id: `id-${i}`,
        model: "m",
        position: 1,
      });
      applyDownloadEvent(state, {
        type: "started",
        id: `id-${i}`,
        files_total: 1,
        bytes_total: 1,
      });
      applyDownloadEvent(state, {
        type: "job_done",
        id: `id-${i}`,
        model: "m",
      });
    }
    expect(state.active).toBeNull();
    expect(state.history.length).toBe(20);
    // Newest first or newest last? We assert newest last (matches server).
    expect(state.history.at(-1)?.id).toBe("id-24");
  });

  it("JobCancelled moves a queued job into terminal history", () => {
    const state = newDownloadsState();
    applyDownloadEvent(state, {
      type: "enqueued",
      id: "a",
      model: "m",
      position: 1,
    });
    applyDownloadEvent(state, { type: "job_cancelled", id: "a" });
    expect(state.queued).toHaveLength(0);
    expect(state.history[0]?.status).toBe("cancelled");
  });
});

import { afterEach, beforeEach, vi } from "vitest";
import { __resetUseDownloadsForTest, useDownloads } from "./useDownloads";

describe("useDownloads.enqueue dispatch", () => {
  const originalFetch = globalThis.fetch;
  const originalEventSource = globalThis.EventSource;

  beforeEach(() => {
    __resetUseDownloadsForTest();
    // Stub EventSource so the singleton's SSE setup is a no-op.
    class StubES {
      onopen: (() => void) | null = null;
      onerror: (() => void) | null = null;
      onmessage: (() => void) | null = null;
      addEventListener() {}
      close() {}
    }
    // @ts-expect-error - test stub
    globalThis.EventSource = StubES;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    globalThis.EventSource = originalEventSource;
    __resetUseDownloadsForTest();
    vi.restoreAllMocks();
  });

  it("routes manifest names to /api/downloads", async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith("/api/downloads")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ id: "job-1", position: 0 }),
          headers: new Map(),
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });
    globalThis.fetch = fetchMock as typeof fetch;

    const dl = useDownloads();
    await dl.enqueue("flux-dev:q4");
    dl.close();

    const urls = fetchMock.mock.calls.map((c) => c[0] as string);
    expect(urls.some((u) => u.endsWith("/api/downloads"))).toBe(true);
    expect(urls.some((u) => u.includes("/api/catalog/"))).toBe(false);
  });

  it("routes cv:/hf: ids to /api/catalog/:id/download", async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes("/api/catalog/")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ primary_job_id: "p", companion_jobs: [] }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });
    globalThis.fetch = fetchMock as typeof fetch;

    const dl = useDownloads();
    await dl.enqueue("cv:232703");
    dl.close();

    const urls = fetchMock.mock.calls.map((c) => c[0] as string);
    expect(
      urls.some((u) => u.includes("/api/catalog/cv%3A232703/download")),
    ).toBe(true);
  });
});
