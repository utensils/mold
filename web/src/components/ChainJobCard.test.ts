import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import ChainJobCard from "./ChainJobCard.vue";
import type { ChainJobDetail } from "../types";

vi.mock("../api", () => ({
  cancelChainJob: vi.fn(),
  chainJobStagePreviewUrl: (jobId: string, stageIdx: number) =>
    `/api/chain-jobs/${jobId}/stages/${stageIdx}/preview`,
  resumeChainJob: vi.fn(),
  retakeChainJob: vi.fn(),
}));

function stage(idx: number) {
  return {
    idx,
    state: "completed" as const,
    seed: `${idx + 1}`,
    frames_emitted: 9,
    generation_time_ms: 10,
    has_preview: false,
    error: null,
  };
}

function job(nextTransition: "smooth" | "cut" | "fade"): ChainJobDetail {
  return {
    id: "job-1",
    state: "failed",
    model: "ltx-2-19b-distilled:fp8",
    stage_count: 2,
    current_stage: 1,
    created_at_unix_ms: 1,
    updated_at_unix_ms: 2,
    error: null,
    ephemeral: false,
    stages: [stage(0), stage(1)],
    finalizes: [],
    retakes: [],
    script: {
      schema: "mold.chain.v1",
      chain: {},
      stage: [
        { prompt: "first", frames: 9, transition: "smooth" },
        { prompt: "second", frames: 9, transition: nextTransition },
      ],
    },
  };
}

describe("ChainJobCard retake controls", () => {
  it("disables splice before a smooth successor with an explanatory title", () => {
    const wrapper = mount(ChainJobCard, {
      props: { job: job("smooth") },
    });

    const splice = wrapper.get('[data-test="chain-job-retake-splice-0"]');
    expect((splice.element as HTMLButtonElement).disabled).toBe(true);
    expect(splice.attributes("title")).toBe(
      "Splice retake requires the next transition to be cut or fade.",
    );
  });

  it.each(["cut", "fade"] as const)(
    "enables splice before a %s successor",
    (transition) => {
      const wrapper = mount(ChainJobCard, {
        props: { job: job(transition) },
      });

      const splice = wrapper.get('[data-test="chain-job-retake-splice-0"]');
      expect((splice.element as HTMLButtonElement).disabled).toBe(false);
      expect(splice.attributes("title")).toBe("Retake only this stage");
    },
  );
});
