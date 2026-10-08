import { describe, expect, it } from "vitest";
import {
  queueDownloadSettlement,
  queueDownloadProgress,
} from "./queueDownloadRecovery";

describe("queue model download feedback", () => {
  it("requires every exact ticket, ignoring stale successes for the same model", () => {
    const jobs = [
      { id: "old", model: "wan", status: "completed" },
      { id: "primary", model: "wan", status: "completed" },
    ];
    expect(queueDownloadSettlement(["new"], jobs)).toEqual({ kind: "waiting" });
    expect(queueDownloadSettlement(["primary", "encoder"], jobs)).toEqual({
      kind: "waiting",
    });
    expect(queueDownloadSettlement(["primary"], jobs)).toEqual({
      kind: "ready",
    });
    expect(queueDownloadSettlement([], jobs)).toEqual({ kind: "waiting" });
    expect(
      queueDownloadSettlement(
        ["primary", "encoder"],
        [
          ...jobs,
          {
            id: "encoder",
            model: "encoder",
            status: "failed",
            error: "Disk full",
          },
        ],
      ),
    ).toEqual({ kind: "failed", message: "Disk full" });
  });
  it("includes completed companions and keeps unknown companion sizes indeterminate", () => {
    const complete = {
      id: "encoder",
      model: "encoder",
      status: "completed",
      bytes_done: 100,
      bytes_total: 100,
    };
    expect(
      queueDownloadProgress(
        [
          complete,
          {
            id: "primary",
            model: "wan",
            status: "active",
            bytes_done: 25,
            bytes_total: 100,
          },
        ],
        "Plato",
      ).fraction,
    ).toBe(0.625);
    expect(
      queueDownloadProgress(
        [complete, { id: "primary", model: "wan", status: "active" }],
        "Plato",
      ).fraction,
    ).toBeNull();
  });
  it("distinguishes queued downloads and unknown totals from measurable progress", () => {
    expect(
      queueDownloadProgress(
        [
          {
            id: "q",
            model: "wan",
            status: "queued",
            bytes_done: 0,
            bytes_total: 0,
          },
        ],
        "Plato",
      ),
    ).toMatchObject({ phase: "queued", fraction: null });
    expect(
      queueDownloadProgress(
        [
          {
            id: "q",
            model: "wan",
            status: "active",
            bytes_done: 25,
            bytes_total: 100,
          },
        ],
        "Plato",
      ),
    ).toMatchObject({ phase: "downloading", fraction: 0.25 });
    expect(queueDownloadProgress([], "Plato")).toMatchObject({
      phase: "reconnecting",
      fraction: null,
    });
  });
});
