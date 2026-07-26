import { flushPromises, mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import MobileSequenceComposer from "./MobileSequenceComposer.vue";
import type { ModelEntry } from "../lib/api/types";

const apiJsonTo = vi.hoisted(() =>
  vi.fn(async () => ({
    model: "ltx-video-0.9.8-2b-distilled:bf16",
    frames_per_clip_cap: 97,
    frames_per_clip_recommended: 97,
    max_stages: 8,
    max_total_frames: 777,
    fade_frames_max: 32,
    transition_modes: ["smooth", "cut", "fade"],
    quantization_family: "bf16",
    supports_audio: false,
  })),
);

vi.mock("../lib/api/client", () => ({ apiJsonTo }));

const model: ModelEntry = {
  name: "ltx-video-0.9.8-2b-distilled:bf16",
  family: "ltx-video",
  downloaded: true,
  is_loaded: false,
  hf_repo: "Lightricks/LTX-Video",
  size_gb: 6,
  default_width: 768,
  default_height: 512,
  default_steps: 7,
  default_guidance: 1,
  description: "",
};

function mountComposer() {
  return mount(MobileSequenceComposer, {
    props: {
      models: [model],
      target: { baseUrl: "http://plato:7680", apiKey: "secret" },
      hostName: "Plato",
      job: null,
      starting: false,
      error: "",
    },
  });
}

describe("MobileSequenceComposer", () => {
  it("starts with two required touch-friendly clip prompts", async () => {
    const wrapper = mountComposer();
    await flushPromises();

    expect(wrapper.findAll("[data-test='mobile-sequence-clip']")).toHaveLength(2);
    expect(
      wrapper
        .findAll("[data-test='mobile-sequence-clip']")
        .map((clip) => clip.get("summary small").text()),
    ).toEqual(["25 frames", "25 frames"]);
    expect(wrapper.get("textarea").attributes("placeholder")).toBe("How does the sequence begin?");
    expect(
      (wrapper.get("[data-test='mobile-generate-sequence']").element as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("submits one reviewed prompt per clip through the durable contract", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const prompts = wrapper.findAll("textarea");
    await prompts[0]!.setValue("A paper boat crosses a moonlit pond");
    await prompts[1]!.setValue("Fireflies gather as the sky brightens");
    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");

    expect(wrapper.emitted("submit")?.[0]?.[0]).toMatchObject({
      model: model.name,
      stages: [
        { prompt: "A paper boat crosses a moonlit pond", frames: 25 },
        { prompt: "Fireflies gather as the sky brightens", frames: 25 },
      ],
      width: 768,
      height: 512,
      output_format: "mp4",
    });
  });

  it("shows one actionable message for a failed GPU-memory job", async () => {
    const wrapper = mount(MobileSequenceComposer, {
      props: {
        models: [model],
        target: { baseUrl: "http://plato:7680", apiKey: "secret" },
        hostName: "Plato",
        starting: false,
        error: "",
        job: {
          id: "failed-sequence",
          state: "failed",
          model: model.name,
          stage_count: 2,
          current_stage: 0,
          created_at_unix_ms: 1,
          updated_at_unix_ms: 2,
          error: 'DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")',
          ephemeral: false,
          stages: [],
          finalizes: [],
          retakes: [],
          script: {
            schema: "mold.chain.v1",
            chain: {
              model: model.name,
              width: 1216,
              height: 704,
              fps: 24,
              steps: 7,
              guidance: 1,
              strength: 1,
              motion_tail_frames: 17,
              output_format: "mp4",
            },
            stage: [],
          },
        },
      },
    });
    await flushPromises();
    const prompts = wrapper.findAll("textarea");
    await prompts[0]!.setValue("Paper boat crosses moonlit pond");
    await prompts[1]!.setValue("Fireflies rise at dawn");

    expect(wrapper.text()).toContain(
      "This sequence needs more GPU memory. Shorten the clip duration or reduce the size, then try again.",
    );
    expect(wrapper.text()).not.toContain("CUDA_ERROR_OUT_OF_MEMORY");
    expect(wrapper.findAll(".mobile-sequence-error")).toHaveLength(1);
  });
});
