import { describe, expect, it } from "vitest";

import {
  LIP_DUB_CONTROL_ID,
  isControlAdapterPipeline,
  normalizeControlId,
  pipelineForControlId,
} from "./ltx2Control";

describe("pipelineForControlId", () => {
  it("routes the lip-dub adapter to its own pipeline", () => {
    expect(pipelineForControlId(LIP_DUB_CONTROL_ID)).toBe("lip-dub");
    expect(pipelineForControlId("LipDub")).toBe("lip-dub");
    expect(pipelineForControlId("  lip_dub  ")).toBe("ic-lora");
  });

  it("leaves every other adapter on the generic in-context pipeline", () => {
    for (const id of ["union", "pose", "detailer", "motion-track", "hdr"]) {
      expect(pipelineForControlId(id)).toBe("ic-lora");
    }
  });
});

describe("normalizeControlId", () => {
  it("matches the server's trim/lowercase/underscore rule", () => {
    expect(normalizeControlId("  Motion_Track ")).toBe("motion-track");
  });
});

describe("isControlAdapterPipeline", () => {
  it("covers both adapter-driven pipelines and nothing else", () => {
    expect(isControlAdapterPipeline("ic-lora")).toBe(true);
    expect(isControlAdapterPipeline("lip-dub")).toBe(true);
    expect(isControlAdapterPipeline("two-stage")).toBe(false);
    expect(isControlAdapterPipeline(null)).toBe(false);
  });
});
