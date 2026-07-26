import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChainJobCard from "./ChainJobCard.vue";
import type { ChainJobDetail } from "../types";

vi.mock("../api", () => ({
  cancelChainJob: vi.fn(),
  chainJobStagePreviewUrl: (jobId: string, stageIdx: number) =>
    `/api/chain-jobs/${jobId}/stages/${stageIdx}/preview`,
  resumeChainJob: vi.fn(),
  retakeChainJob: vi.fn(),
}));

const { cancelChainJob } = await import("../api");

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
  beforeEach(() => {
    vi.mocked(cancelChainJob).mockReset();
    vi.mocked(cancelChainJob).mockResolvedValue({} as never);
  });

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

  it("renders the actionable job and stage failure instead of only saying failed", () => {
    const failed = job("smooth");
    failed.error =
      "render-chain v1 supports one-stage and distilled LTX-2 pipelines, got TwoStage";
    failed.stages[0]!.state = "failed";
    failed.stages[0]!.error = "TwoStage is not supported for sequences";
    const wrapper = mount(ChainJobCard, { props: { job: failed } });

    expect(wrapper.get("[data-test='chain-job-error']").text()).toContain(
      "distilled LTX-2",
    );
    expect(
      wrapper.get("[data-test='chain-job-stage-error-0']").text(),
    ).toContain("TwoStage");
  });

  it("cancels queued work once and disables the control while awaiting the server", async () => {
    let resolveCancel!: () => void;
    vi.mocked(cancelChainJob).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveCancel = () => resolve({} as never);
        }),
    );
    const queued = job("cut");
    queued.state = "queued";
    const wrapper = mount(ChainJobCard, { props: { job: queued } });

    await wrapper.get("[data-test='chain-job-cancel']").trigger("click");
    expect(cancelChainJob).toHaveBeenCalledTimes(1);
    expect(wrapper.get("[data-test='chain-job-cancel']").text()).toContain(
      "Cancelling",
    );
    expect(
      (
        wrapper.get("[data-test='chain-job-cancel']")
          .element as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    await wrapper.get("[data-test='chain-job-cancel']").trigger("click");
    expect(cancelChainJob).toHaveBeenCalledTimes(1);

    resolveCancel();
    await wrapper.vm.$nextTick();
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("updated")).toBeTruthy();
    expect(
      (
        wrapper.get("[data-test='chain-job-cancel']")
          .element as HTMLButtonElement
      ).disabled,
    ).toBe(false);
  });

  it("surfaces cancel failures and leaves settled jobs with a dismiss action", async () => {
    vi.mocked(cancelChainJob).mockRejectedValueOnce(
      new Error("cancel request failed"),
    );
    const running = job("cut");
    running.state = "running";
    const wrapper = mount(ChainJobCard, { props: { job: running } });
    await wrapper.get("[data-test='chain-job-cancel']").trigger("click");
    await wrapper.vm.$nextTick();
    expect(
      wrapper.get("[data-test='chain-job-action-error']").text(),
    ).toContain("cancel request failed");

    const failedWrapper = mount(ChainJobCard, { props: { job: job("cut") } });
    expect(failedWrapper.find("[data-test='chain-job-cancel']").exists()).toBe(
      false,
    );
    await failedWrapper.get("[data-test='chain-job-dismiss']").trigger("click");
    expect(failedWrapper.emitted("dismiss")).toBeTruthy();

    const interrupted = job("cut");
    interrupted.state = "interrupted";
    const interruptedWrapper = mount(ChainJobCard, {
      props: { job: interrupted },
    });
    expect(
      interruptedWrapper.find("[data-test='chain-job-resume']").exists(),
    ).toBe(true);
    expect(
      interruptedWrapper.find("[data-test='chain-job-dismiss']").exists(),
    ).toBe(false);
  });
});
