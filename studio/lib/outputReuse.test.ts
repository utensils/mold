import { describe, expect, it } from "vitest";
import { pipelineForSettingsReuse } from "./outputReuse";

describe("pipelineForSettingsReuse", () => {
  it("does not turn an automatic chain's runtime pipeline into an authored override", () => {
    expect(
      pipelineForSettingsReuse({
        pipeline: "distilled",
        output_mode: "one-shot",
        chain: { stages: [{ frames: 97 }, { frames: 97 }] },
      }),
    ).toBeNull();
    expect(
      pipelineForSettingsReuse({
        pipeline: "distilled",
        chain: { stages: [{ frames: 97 }, { frames: 97 }] },
      }),
    ).toBeNull();
    expect(
      pipelineForSettingsReuse({
        pipeline: "distilled",
        output_mode: "one-shot",
      }),
    ).toBeNull();
    expect(pipelineForSettingsReuse({ pipeline: "distilled" })).toBeNull();
  });

  it("preserves a pipeline only when request provenance says it was authored", () => {
    expect(
      pipelineForSettingsReuse({
        pipeline: "two-stage",
        output_mode: "one-shot",
        pipeline_requested: true,
      }),
    ).toBe("two-stage");
    expect(
      pipelineForSettingsReuse({
        pipeline: "distilled",
        output_mode: "one-shot",
        pipeline_requested: false,
      }),
    ).toBeNull();
    expect(
      pipelineForSettingsReuse({
        pipeline: "distilled",
        chain_job_id: "legacy-authored-sequence",
        chain: { stages: [] },
      }),
    ).toBe("distilled");
  });
});
