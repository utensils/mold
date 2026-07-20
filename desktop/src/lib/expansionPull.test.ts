import { describe, expect, it } from "vitest";
import { resolveExpansionPullStatus } from "./expansionPull";
import type { DownloadJob } from "./api/types";

function job(
  id: string,
  status: DownloadJob["status"],
  overrides: Partial<DownloadJob> = {},
): DownloadJob {
  return {
    id,
    model: "qwen3-expand:q8",
    status,
    files_done: 0,
    files_total: 2,
    bytes_done: 0,
    bytes_total: 2_000,
    ...overrides,
  };
}

describe("resolveExpansionPullStatus", () => {
  it("follows the returned job id through queued, pulling, and ready states", () => {
    const base = {
      model: "qwen3-expand:q8",
      phase: "starting" as const,
      jobId: "pull-2",
      baselineJobIds: ["old-pull"],
      allowExistingInFlight: false,
      activeJobs: [] as DownloadJob[],
      queued: [job("pull-2", "queued")],
      history: [job("old-pull", "completed")],
    };

    expect(resolveExpansionPullStatus(base)).toMatchObject({
      kind: "queued",
      job: { id: "pull-2" },
    });
    expect(
      resolveExpansionPullStatus({
        ...base,
        queued: [],
        activeJobs: [
          job("pull-2", "active", {
            bytes_done: 750,
            current_file: "text_encoder/model.safetensors",
          }),
        ],
      }),
    ).toMatchObject({ kind: "pulling", job: { bytes_done: 750 } });
    expect(
      resolveExpansionPullStatus({
        ...base,
        queued: [],
        history: [job("pull-2", "completed"), job("old-pull", "completed")],
      }),
    ).toMatchObject({ kind: "ready", job: { id: "pull-2" } });
  });

  it("matches a new exact-model snapshot that arrives before an older server returns no id", () => {
    const status = resolveExpansionPullStatus({
      model: "qwen3-expand:q8",
      phase: "starting",
      jobId: null,
      baselineJobIds: ["unrelated", "stale-same-model"],
      allowExistingInFlight: false,
      activeJobs: [],
      queued: [
        job("unrelated", "queued", { model: "flux-dev:q8" }),
        job("stale-same-model", "queued"),
        job("snapshot-before-response", "queued"),
      ],
      history: [],
    });

    expect(status).toMatchObject({ kind: "queued", job: { id: "snapshot-before-response" } });
  });

  it("matches the real bare expansion model to its canonical tagged download row", () => {
    const status = resolveExpansionPullStatus({
      model: "qwen3-expand",
      phase: "starting",
      jobId: null,
      baselineJobIds: [],
      allowExistingInFlight: false,
      activeJobs: [],
      queued: [job("canonical-row", "queued", { model: "qwen3-expand:q8" })],
      history: [],
    });

    expect(status).toMatchObject({ kind: "queued", job: { id: "canonical-row" } });
  });

  it("never mistakes stale same-model history for the new pull", () => {
    const status = resolveExpansionPullStatus({
      model: "qwen3-expand:q8",
      phase: "starting",
      jobId: null,
      baselineJobIds: ["stale-same-model"],
      allowExistingInFlight: false,
      activeJobs: [],
      queued: [],
      history: [job("stale-same-model", "completed")],
    });

    expect(status).toEqual({ kind: "starting", job: null });
  });

  it("can adopt an already-running exact-model job after a 409", () => {
    const status = resolveExpansionPullStatus({
      model: "qwen3-expand:q8",
      phase: "starting",
      jobId: null,
      baselineJobIds: ["already-running"],
      allowExistingInFlight: true,
      activeJobs: [job("already-running", "active")],
      queued: [],
      history: [],
    });

    expect(status).toMatchObject({ kind: "pulling", job: { id: "already-running" } });
  });

  it.each(["jobId", "observedJobId"] as const)(
    "waits only for authoritative %s A instead of following competing same-model B",
    (field) => {
      const status = resolveExpansionPullStatus({
        model: "qwen3-expand",
        phase: "starting",
        jobId: field === "jobId" ? "pull-A" : null,
        observedJobId: field === "observedJobId" ? "pull-A" : null,
        baselineJobIds: [],
        allowExistingInFlight: false,
        activeJobs: [],
        queued: [job("pull-B", "queued", { model: "qwen3-expand:q8" })],
        history: [],
      });

      expect(status).toEqual({ kind: "starting", job: null });
    },
  );

  it.each(["failed", "cancelled"] as const)("surfaces %s as directed recovery", (state) => {
    const status = resolveExpansionPullStatus({
      model: "qwen3-expand:q8",
      phase: "starting",
      jobId: "pull-2",
      baselineJobIds: [],
      allowExistingInFlight: false,
      activeJobs: [],
      queued: [],
      history: [job("pull-2", state, { error: state === "failed" ? "disk full" : null })],
    });

    expect(status).toMatchObject({ kind: state, job: { id: "pull-2" } });
  });
});
