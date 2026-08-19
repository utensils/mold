import { describe, expect, test } from "vitest";
import {
  classifyMissingModel,
  type GenerationPlacementPreview,
} from "./generationPlacement";

function infeasible(
  reason: string,
  missing: GenerationPlacementPreview["missing_components"] = [],
): GenerationPlacementPreview {
  return {
    version: 1,
    authoritative: true,
    state_version: 3,
    plan_version: 9,
    outcome: "infeasible",
    reason,
    missing_components: missing,
  };
}

describe("classifyMissingModel", () => {
  test("names the requested model when the primary artifacts are absent", () => {
    const preview = infeasible(
      "model 'z-image-turbo:q6' has no concrete local artifacts",
      [
        {
          kind: "transformer",
          name: "transformer",
          present: false,
          repair_model: "z-image-turbo:q6",
        },
      ],
    );
    expect(classifyMissingModel(preview, "z-image-turbo:q6")).toEqual({
      model: "z-image-turbo:q6",
      missingComponents: [
        {
          kind: "transformer",
          name: "transformer",
          present: false,
          repair_model: "z-image-turbo:q6",
        },
      ],
    });
  });

  test("reads the reason alone when the component listing could not be built", () => {
    const preview = infeasible(
      "model 'cv:1759168' has no concrete local artifacts",
    );
    expect(classifyMissingModel(preview, "cv:1759168")).toEqual({
      model: "cv:1759168",
      missingComponents: [],
    });
  });

  test("reads a missing primary component even when the reason is worded differently", () => {
    const preview = infeasible("this model is not installed here", [
      {
        kind: "checkpoint",
        name: "primary checkpoint",
        present: false,
        repair_model: "hf:org/repo",
      },
    ]);
    expect(classifyMissingModel(preview, "hf:org/repo")?.model).toBe(
      "hf:org/repo",
    );
  });

  test("is not a missing model when the machine simply cannot fit it", () => {
    expect(
      classifyMissingModel(
        infeasible(
          "no device can host this generation: needs 48.0 GB, largest device has 24.0 GB",
        ),
        "flux-dev:bf16",
      ),
    ).toBeNull();
  });

  test("is not a missing model when only a companion component is absent", () => {
    const preview = infeasible("text encoder t5-fp16 is not installed", [
      {
        kind: "text-encoder",
        name: "t5xxl",
        present: false,
        repair_model: "t5-fp16",
      },
    ]);
    expect(classifyMissingModel(preview, "flux-dev:q8")).toBeNull();
  });

  test("ignores components the server reported as present", () => {
    const preview = infeasible("something else went wrong", [
      {
        kind: "transformer",
        name: "transformer",
        present: true,
        repair_model: "flux-dev:q8",
      },
    ]);
    expect(classifyMissingModel(preview, "flux-dev:q8")).toBeNull();
  });

  test("requires an authoritative infeasible preview", () => {
    const planned: GenerationPlacementPreview = {
      version: 1,
      authoritative: true,
      state_version: 1,
      plan_version: 1,
      outcome: "planned",
      candidate: {
        device_id: "cuda:0",
        execution_fingerprint: "exec",
        predicted_start_after_ms: 0,
        predicted_completion_after_ms: 10,
        setup_ms: 0,
        setup_kind: "warm",
        estimate_confidence: "high",
      },
    };
    expect(classifyMissingModel(planned, "flux-dev:q8")).toBeNull();
    expect(
      classifyMissingModel(
        {
          ...infeasible("model 'x' has no concrete local artifacts"),
          authoritative: false,
        },
        "x",
      ),
    ).toBeNull();
    expect(classifyMissingModel(null, "flux-dev:q8")).toBeNull();
    expect(
      classifyMissingModel(
        infeasible("model '' has no concrete local artifacts"),
        "",
      ),
    ).toBeNull();
  });
});
