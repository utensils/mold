import { mount } from "@vue/test-utils";
import { expect, it } from "vitest";
import VideoDurationEditor from "./VideoDurationEditor.vue";
import type { VideoDurationAuthoring } from "../lib/videoDurationAuthoring";
const initial: VideoDurationAuthoring = {
  durationMs: "30000",
  scenesEnabled: false,
  scenes: [],
};
it("keeps scenes optional and hands the first edited scene back to the ordinary prompt", async () => {
  const wrapper = mount(VideoDurationEditor, {
    props: { modelValue: initial, prompt: "Original prompt" },
  });
  expect(wrapper.findAll("textarea")).toHaveLength(0);
  await wrapper.get('[data-test="define-video-scenes"]').trigger("click");
  const draft = wrapper.emitted(
    "update:modelValue",
  )![0]![0] as VideoDurationAuthoring;
  expect(draft.scenes[0]?.prompt).toBe("Original prompt");
  await wrapper.setProps({
    modelValue: {
      ...draft,
      scenes: [{ ...draft.scenes[0]!, prompt: "Edited scene" }],
    },
  });
  await wrapper.get('[data-test="video-one-prompt"]').trigger("click");
  expect(wrapper.emitted("singlePrompt")![0]).toEqual(["Edited scene"]);
  expect(wrapper.findAll('button[type="submit"]')).toHaveLength(0);
});
it("sums scene times exactly and supports ordered editing without hiding an invalid row", async () => {
  const wrapper = mount(VideoDurationEditor, {
    props: {
      modelValue: {
        ...initial,
        scenesEnabled: true,
        scenes: [
          { id: "a", prompt: "First", durationMs: "1000" },
          { id: "b", prompt: "Second", durationMs: "2000" },
        ],
      },
      prompt: "Original",
    },
  });
  await wrapper.get('[data-test="video-scene-a-up"]').trigger("click");
  expect(wrapper.emitted("update:modelValue")).toBeUndefined();
  await wrapper.get('[data-test="video-scene-b-up"]').trigger("click");
  const moved = wrapper
    .emitted("update:modelValue")!
    .at(-1)![0] as VideoDurationAuthoring;
  expect(moved.scenes.map((scene) => scene.prompt)).toEqual([
    "Second",
    "First",
  ]);
  expect(moved.durationMs).toBe("3000");
  await wrapper.get('[data-test="video-scene-a-prompt"]').setValue("");
  const invalid = wrapper
    .emitted("update:modelValue")!
    .at(-1)![0] as VideoDurationAuthoring;
  expect(invalid.scenes).toHaveLength(2);
  expect(invalid.scenes[0]?.prompt).toBe("");
});
it("discloses that reusing an extended movie makes fresh footage", () => {
  const w = mount(VideoDurationEditor, {
    props: {
      modelValue: {
        ...initial,
        reuseNotice:
          "These settings make new footage. The original video is not attached.",
      },
      prompt: "new footage",
    },
  });
  expect(w.get('[role="status"]').text()).toContain(
    "original video is not attached",
  );
});

const plan = {
  validation_revision: "r",
  model: "minimax-h3",
  width: 768,
  height: 432,
  fps: 24,
  single_clip_max_frames: 345,
  overlap_frames: 1,
  join_mode: "continuous" as const,
  clip_count: "3",
  part_count: "1",
  added_frames: "960",
  source_frames: "0",
  total_frames: "960",
  added_duration_ms: "40000",
  total_duration_ms: "40000",
  warnings: [
    "Scene 2 is 40 s but one generation is 14.4 s, so its prompt is reused for 3 clips and the action will repeat. Split it into scenes of 14 s or less.",
    "Scene 3 is 1 s, shorter than the shortest generation, so it is rendered longer and trimmed.",
  ],
};
it("states the one-generation limit, the one-clip scene length and every warning in full", () => {
  const wrapper = mount(VideoDurationEditor, {
    props: {
      modelValue: {
        durationMs: "40000",
        scenesEnabled: true,
        scenes: [
          { id: "a", prompt: "one", durationMs: "20000" },
          { id: "b", prompt: "two", durationMs: "20000" },
        ],
      },
      prompt: "",
      plan,
    },
  });
  const text = wrapper.text();
  expect(text).toContain("Single generation: up to 14.375 seconds");
  expect(text).toContain("Scenes up to 14.3 s each stay one clip");
  for (const warning of plan.warnings) {
    const row = wrapper
      .findAll('[data-test="video-plan-warning"]')
      .find((node) => node.text() === warning);
    expect(row, warning).toBeTruthy();
  }
});
it("keeps the one-prompt duration on the shared single-generation sentence", () => {
  const wrapper = mount(VideoDurationEditor, {
    props: { modelValue: initial, prompt: "", plan },
  });
  expect(wrapper.text()).toContain("Single generation: up to 14.375 seconds");
  expect(wrapper.text()).not.toContain("stay one clip");
  expect(wrapper.findAll('[data-test="video-plan-warning"]')).toHaveLength(2);
});
