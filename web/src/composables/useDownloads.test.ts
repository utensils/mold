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
