import { describe, expect, it } from "vitest";
import {
  pipelineForSettingsReuse,
  transparentBackgroundForSettingsReuse,
} from "./outputReuse";

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

describe("transparentBackgroundForSettingsReuse", () => {
  it("restores the toggle a transparent print was made with", () => {
    expect(
      transparentBackgroundForSettingsReuse({ transparent_background: true }),
    ).toBe(true);
  });

  it("leaves it off for every other print, including an alpha reference edit", () => {
    // `has_alpha` is a fact about the FILE (a transparent reference keeps its
    // alpha with the toggle off), not a setting the user chose.
    expect(transparentBackgroundForSettingsReuse({ has_alpha: true })).toBe(
      false,
    );
    expect(transparentBackgroundForSettingsReuse({})).toBe(false);
    expect(
      transparentBackgroundForSettingsReuse({ transparent_background: null }),
    ).toBe(false);
  });
});
