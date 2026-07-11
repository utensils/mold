import { describe, expect, it } from "vitest";
import { applyDownloadEvent, emptyDownloadsState } from "./downloads";
import type { DownloadEvent, DownloadsListing } from "../lib/api/types";

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
