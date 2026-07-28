import { describe, expect, it } from "vitest";
import { isPrintOfChainJob } from "./sequenceReuse";

describe("isPrintOfChainJob", () => {
  it("matches the print a durable sequence produced", () => {
    expect(isPrintOfChainJob({ chain_job_id: "job-1" }, "job-1")).toBe(true);
  });

  it("rejects a row that merely looks similar", () => {
    // A legacy or ephemeral chain output carries no job id; matching it would
    // hand the canvas somebody else's video.
    expect(isPrintOfChainJob({}, "job-1")).toBe(false);
    expect(isPrintOfChainJob({ chain_job_id: null }, "job-1")).toBe(false);
    expect(isPrintOfChainJob({ chain_job_id: "job-2" }, "job-1")).toBe(false);
    expect(isPrintOfChainJob({ chain_job_id: "job-1" }, "")).toBe(false);
  });
});
