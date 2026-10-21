import { mount } from "@vue/test-utils";
import { expect, it } from "vitest";
import VideoDurationInput from "./VideoDurationInput.vue";

it("shows the server safe limit and actual joined clip count with one duration field", async () => {
  const wrapper = mount(VideoDurationInput, {
    props: {
      modelValue: "30000",
      plan: { fps: 24, single_clip_max_frames: 107, clip_count: "7" },
    },
  });
  expect(wrapper.text()).toContain("Single generation: up to 4.458 seconds");
  expect(wrapper.text()).toContain("7 clips joined into one video");
  const input = wrapper.get('input[aria-label="Duration in seconds"]');
  expect(input.attributes("max")).toBeUndefined();
  await input.setValue("3600");
  expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual(["3600000"]);
});
it("keeps invalid custom text visible and reports it inline", async () => {
  const wrapper = mount(VideoDurationInput, { props: { modelValue: "30000" } });
  await wrapper.get("input").setValue("1e9");
  expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual([null]);
  expect(wrapper.get('[role="alert"]').text()).toContain("seconds");
  expect((wrapper.get("input").element as HTMLInputElement).value).toBe("1e9");
  expect(wrapper.text()).not.toContain("Single generation:");
});
it("never rounds very large server clip counts or claims an absent plan", () => {
  const wrapper = mount(VideoDurationInput, {
    props: {
      modelValue: "30000",
      plan: {
        fps: 24,
        single_clip_max_frames: 107,
        clip_count: "9007199254740993",
      },
    },
  });
  expect(wrapper.text()).toContain(
    "9007199254740993 clips joined into one video",
  );
});
