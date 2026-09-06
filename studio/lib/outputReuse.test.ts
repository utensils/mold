import { describe, expect, it } from "vitest";
import { pipelineForSettingsReuse } from "./outputReuse";

describe("pipelineForSettingsReuse", () => {
  it("does not turn a runtime-resolved pipeline into an authored override", () => {
    expect(pipelineForSettingsReuse({ pipeline: "distilled" })).toBeNull();
    expect(
      pipelineForSettingsReuse({
        pipeline: "distilled",
        pipeline_requested: false,
      }),
    ).toBeNull();
  });

  it("preserves a pipeline only when request provenance says it was authored", () => {
    expect(
      pipelineForSettingsReuse({
        pipeline: "two-stage",
        pipeline_requested: true,
      }),
    ).toBe("two-stage");
  });

  // Scene authoring is retired, so every reuse rebuilds a one-shot. A print
  // stitched from a scripted sequence must not hand its recorded pipeline to
  // that form: pinning it is exactly what disables the automatic chain route
  // and collapses the duration control to "1 generation".
  it("gives a sequence-stitched print's pipeline back to Auto", () => {
    expect(
      pipelineForSettingsReuse({
        pipeline: "distilled",
        pipeline_requested: null,
      }),
    ).toBeNull();
  });
});
