import { describe, expect, test } from "vitest";
import {
  pullResumeFailureMessage,
  resolvePullResumeOutcome,
  terminalPullJobIds,
  type PullResumeJob,
} from "./pullResume";

const job = (over: Partial<PullResumeJob> = {}): PullResumeJob => ({
  id: "job-1",
  model: "z-image-turbo:q6",
  status: "completed",
  ...over,
});

describe("resolvePullResumeOutcome", () => {
  test("waits while the watched job is still running", () => {
    expect(
      resolvePullResumeOutcome([job({ status: "downloading" })], {
        model: "z-image-turbo:q6",
        jobId: "job-1",
        seenTerminal: [],
      }),
    ).toEqual({ kind: "waiting" });
  });

  test("resumes on the exact watched job", () => {
    const done = job();
    expect(
      resolvePullResumeOutcome([done], {
        model: "z-image-turbo:q6",
        jobId: "job-1",
        seenTerminal: [],
      }),
    ).toEqual({ kind: "ready", job: done });
  });

  test("never resumes off a stale completed pull of the same model", () => {
    const stale = job({ id: "old-job" });
    expect(
      resolvePullResumeOutcome([stale], {
        model: "z-image-turbo:q6",
        jobId: "job-1",
        seenTerminal: [],
      }),
    ).toEqual({ kind: "waiting" });
    // Without a job id the pre-existing terminal row is excluded by name.
    expect(
      resolvePullResumeOutcome([stale], {
        model: "z-image-turbo:q6",
        jobId: null,
        seenTerminal: ["old-job"],
      }),
    ).toEqual({ kind: "waiting" });
  });

  test("matches by model when the server reported no job id", () => {
    const done = job({ id: "new-job" });
    expect(
      resolvePullResumeOutcome([done], {
        model: "z-image-turbo:q6",
        jobId: null,
        seenTerminal: ["old-job"],
      }),
    ).toEqual({ kind: "ready", job: done });
  });

  test("reports a failed or cancelled pull instead of resuming", () => {
    const failed = job({ status: "failed", error: "disk full" });
    expect(
      resolvePullResumeOutcome([failed], {
        model: "z-image-turbo:q6",
        jobId: "job-1",
        seenTerminal: [],
      }),
    ).toEqual({ kind: "failed", job: failed });
    expect(pullResumeFailureMessage("z-image-turbo:q6", failed)).toBe(
      "Download of z-image-turbo:q6 failed — disk full; generation not resumed.",
    );
    expect(
      pullResumeFailureMessage(
        "z-image-turbo:q6",
        job({ status: "cancelled" }),
      ),
    ).toBe(
      "Download of z-image-turbo:q6 was cancelled; generation not resumed.",
    );
  });

  test("terminalPullJobIds snapshots only settled rows", () => {
    expect(
      terminalPullJobIds([
        job({ id: "a", status: "completed" }),
        job({ id: "b", status: "downloading" }),
        job({ id: "c", status: "cancelled" }),
      ]),
    ).toEqual(["a", "c"]);
  });
});
