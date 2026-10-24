import { describe, expect, it } from "vitest";
import fixtures from "../../docs/contracts/video-job-rows.json";
import type { ActiveWorkItem } from "../api/activity";
import type { VideoJob } from "./api/videoJobTypes";
import {
  isLiveVideoJobActivity,
  liveVideoJobCounts,
  videoJobActivityItem,
  videoJobActivityRow,
  videoJobRow,
  videoJobRowFromJob,
} from "./videoJobRow";

describe("cross-surface video job row contract", () => {
  for (const fixture of fixtures) {
    it(`${fixture.input.state} ${fixture.input.progress_phase ?? "-"} ${fixture.input.completed_clips}/${fixture.input.clip_count} → ${fixture.row.caption}`, () => {
      expect(
        videoJobRow(fixture.input as Parameters<typeof videoJobRow>[0]),
      ).toEqual(fixture.row);
    });
  }

  it("never says 1 clips", () => {
    for (const state of ["queued", "running", "paused"] as const) {
      const row = videoJobRow({
        state,
        completed_clips: "0",
        clip_count: "1",
        total_duration_ms: "5000",
        can_cancel: true,
        current: true,
      });
      expect(row.caption).not.toContain("clips");
      expect(row.detail).not.toContain("clips");
    }
  });

  it("reads an unknown phase from a newer server as rendering", () => {
    expect(
      videoJobRow({
        state: "running",
        progress_phase: "uploading",
        completed_clips: "1",
        clip_count: "3",
        total_duration_ms: "40000",
        can_cancel: true,
        current: true,
      }).caption,
    ).toBe("Rendering clip 2 of 3");
  });

  it("reads a VideoJob", () => {
    const job = {
      state: "running",
      progress_phase: "joining",
      completed_clips: "3",
      can_cancel: true,
      plan: { clip_count: "3", total_duration_ms: "40000" },
    } as unknown as VideoJob;
    expect(videoJobRowFromJob(job).caption).toBe("Joining clips");
    expect(videoJobRowFromJob(job, false).caption).toBe(
      "Last known: Joining clips",
    );
  });
});

describe("video job activity rows", () => {
  const item = (patch: Partial<ActiveWorkItem>): ActiveWorkItem => ({
    id: "v1",
    kind: "video_job",
    execution: "video_job",
    phase: "running",
    created_at_unix_ms: 1,
    updated_at_unix_ms: 1,
    current: 0,
    total: 3,
    can_cancel: true,
    ...patch,
  });

  it.each([
    [{ phase: "queued", current: 0 }, "Waiting", null],
    [{ phase: "queued", current: 0, total: 1 }, "Waiting", null],
    [{ phase: "running", current: 1 }, "Rendering clip 2 of 3", 1 / 3],
    [{ phase: "running", current: 0, total: 1 }, "Rendering", null],
    [{ phase: "running", current: 3 }, "Joining clips", 1],
    [{ phase: "running", current: 1, total: 1 }, "Finishing", null],
    [{ phase: "cancelling", can_cancel: false }, "Stopping", null],
    [
      { phase: "paused", current: 1, can_cancel: false },
      "Paused · 1 of 3 clips done",
      null,
    ],
    [
      { phase: "interrupted", current: 2, total: 4 },
      "Interrupted · 2 of 4 clips done",
      null,
    ],
    [{ phase: "failed" }, "Failed", null],
    [{ phase: "cancelled" }, "Stopped", null],
    [{ phase: "completed", current: 3 }, "Done", null],
  ] as const)("%o reads %s", (patch, caption, progress) => {
    const row = videoJobActivityRow(item(patch));
    expect(row.caption).toBe(caption);
    expect(row.progress).toBe(progress);
  });

  it("counts only unsettled parents as live", () => {
    expect(isLiveVideoJobActivity(item({ phase: "running" }))).toBe(true);
    expect(isLiveVideoJobActivity(item({ phase: "queued" }))).toBe(true);
    expect(isLiveVideoJobActivity(item({ phase: "cancelling" }))).toBe(true);
    for (const phase of [
      "paused",
      "failed",
      "interrupted",
      "cancelled",
      "completed",
    ])
      expect(isLiveVideoJobActivity(item({ phase }))).toBe(false);
    expect(isLiveVideoJobActivity(item({ kind: "generation" }))).toBe(false);
  });
});

it("counts live video parents as running or waiting for a queue summary", () => {
  const row = (phase: string, kind = "video_job") =>
    ({ kind, phase, current: 0, total: 2 }) as ActiveWorkItem;
  expect(
    liveVideoJobCounts([
      row("running"),
      row("cancelling"),
      row("queued"),
      row("paused"),
      row("failed"),
      row("running", "generation"),
    ]),
  ).toEqual({ running: 2, waiting: 1 });
});

it("words a saved job and its activity projection identically", () => {
  for (const fixture of fixtures) {
    if (!fixture.input.current) continue;
    const job = {
      id: "j",
      state: fixture.input.state,
      progress_phase: fixture.input.progress_phase,
      completed_clips: fixture.input.completed_clips,
      can_cancel: fixture.input.can_cancel,
      created_at_ms: 1,
      updated_at_ms: 2,
      plan: {
        model: "m",
        clip_count: fixture.input.clip_count,
        total_duration_ms: fixture.input.total_duration_ms,
      },
    } as unknown as VideoJob;
    const item = videoJobActivityItem(job);
    expect(item.kind).toBe("video_job");
    expect(
      videoJobActivityRow(item).caption,
      JSON.stringify(fixture.input),
    ).toBe(fixture.row.caption);
  }
});
