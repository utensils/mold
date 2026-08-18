import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import VideoDurationSlider from "./VideoDurationSlider.vue";

describe("VideoDurationSlider", () => {
  it("shows seconds and emits model-valid frames", async () => {
    const wrapper = mount(VideoDurationSlider, {
      props: {
        frames: 97,
        fps: 24,
        model: {
          max_runtime_seconds: 20,
          max_frames_absolute: 604,
          frame_step: 8,
        },
      },
    });

    expect(wrapper.text()).toContain("4.0s");
    expect(wrapper.text()).toContain("1 generation");
    const range = wrapper.get('input[type="range"]');
    expect(range.attributes("max")).toBe("481");
    await range.setValue("241");
    expect(wrapper.emitted("update:frames")?.[0]).toEqual([241]);
  });

  it("shows shared automatic-generation notches and count", () => {
    const wrapper = mount(VideoDurationSlider, {
      props: {
        frames: 241,
        fps: 24,
        model: {
          name: "ltx-2-19b-distilled:fp8",
          family: "ltx2",
          max_runtime_seconds: 20,
          max_frames_absolute: 604,
          frame_step: 8,
        },
      },
    });

    expect(wrapper.text()).toContain("3 generations");
    expect(wrapper.text()).toContain("automatic sequence");
    expect(
      wrapper.get('input[type="range"]').attributes("aria-valuetext"),
    ).toBe("10s, 3 generations");
    expect(
      wrapper.findAll(".ms-slider__mark b").map((mark) => mark.text()),
    ).toEqual(["1×", "2×", "3×", "4×", "5×", "6×"]);
  });

  it("follows submit-time temporal-upscale routing without an invisible snap", async () => {
    const wrapper = mount(VideoDurationSlider, {
      props: {
        frames: 241,
        fps: 24,
        model: {
          name: "ltx-2-19b-distilled:fp8",
          family: "ltx2",
          max_runtime_seconds: 20,
          max_frames_absolute: 604,
          frame_step: 8,
        },
        routingRequest: { temporal_upscale: "x2" },
      },
    });

    expect(wrapper.text()).toContain("1 generation");
    expect(wrapper.text()).not.toContain("automatic sequence");
    expect(
      wrapper.findAll(".ms-slider__mark b").map((mark) => mark.text()),
    ).toEqual([]);
    await wrapper.get('input[type="range"]').trigger("pointerdown");
    await wrapper.get('input[type="range"]').setValue("473");
    await wrapper.get('input[type="range"]').trigger("pointerup");
    expect(wrapper.emitted("update:frames")?.at(-1)).toEqual([473]);
  });

  it("magnetically snaps near a generation boundary while retaining fine steps", async () => {
    const wrapper = mount(VideoDurationSlider, {
      props: {
        frames: 241,
        fps: 24,
        model: {
          name: "ltx-2-19b-distilled:fp8",
          family: "ltx2",
          max_runtime_seconds: 20,
          max_frames_absolute: 604,
          frame_step: 8,
        },
      },
    });

    const range = wrapper.get('input[type="range"]');
    await range.trigger("pointerdown");
    await range.setValue("249");
    await range.trigger("pointerup");
    expect(wrapper.emitted("update:frames")?.at(-1)).toEqual([257]);
    await range.setValue("225");
    expect(wrapper.emitted("update:frames")?.at(-1)).toEqual([225]);
  });

  it("uses a wider magnetic capture area for touch drags", async () => {
    const wrapper = mount(VideoDurationSlider, {
      props: {
        frames: 241,
        fps: 24,
        touchFriendly: true,
        model: {
          name: "ltx-2-19b-distilled:fp8",
          family: "ltx2",
          max_runtime_seconds: 20,
          max_frames_absolute: 604,
          frame_step: 8,
        },
      },
    });

    const range = wrapper.get('input[type="range"]');
    await range.trigger("pointerdown");
    await range.setValue("193");
    await range.trigger("pointerup");
    expect(wrapper.emitted("update:frames")?.at(-1)).toEqual([177]);
  });

  it("lets keyboard input move one frame-grid step away from a notch", async () => {
    const wrapper = mount(VideoDurationSlider, {
      props: {
        frames: 97,
        fps: 24,
        model: {
          name: "ltx-2-19b-distilled:fp8",
          family: "ltx2",
          max_runtime_seconds: 20,
          max_frames_absolute: 604,
          frame_step: 8,
        },
      },
    });

    await wrapper.get('input[type="range"]').setValue("105");
    expect(wrapper.emitted("update:frames")?.at(-1)).toEqual([105]);
  });

  it("updates the request value when FPS lowers the model ceiling", async () => {
    const wrapper = mount(VideoDurationSlider, {
      props: {
        frames: 481,
        fps: 24,
        model: {
          max_runtime_seconds: 20,
          max_frames_absolute: 604,
          frame_step: 8,
        },
      },
    });

    await wrapper.setProps({ fps: 12 });
    expect(wrapper.emitted("update:frames")?.at(-1)).toEqual([241]);
    await wrapper.setProps({ frames: 241 });
    expect(wrapper.text()).toContain("241 frames");
  });

  it("preserves the exact one-frame value accepted by the server", () => {
    const wrapper = mount(VideoDurationSlider, {
      props: { frames: 1, fps: 24, model: { frame_step: 8 } },
    });

    expect(wrapper.get('input[type="range"]').attributes("min")).toBe("1");
    expect(wrapper.emitted("update:frames")).toBeUndefined();
  });

  it("uses the advertised minimum and offset for H3's frame grid", () => {
    const wrapper = mount(VideoDurationSlider, {
      props: {
        frames: 124,
        fps: 24,
        model: {
          family: "minimax-h3",
          min_frames: 124,
          max_frames: 362,
          frame_step: 17,
          frame_offset: 5,
        },
      },
    });

    const slider = wrapper.get('input[type="range"]');
    expect(slider.attributes("min")).toBe("124");
    expect(slider.attributes("max")).toBe("362");
    expect(slider.attributes("step")).toBe("17");
  });

  it("reports an intentional long-chain duration without rewriting it", () => {
    const wrapper = mount(VideoDurationSlider, {
      props: {
        frames: 500,
        fps: 12,
        model: { family: "ltx2", frame_step: 8 },
      },
    });

    expect(wrapper.text()).toContain("500 frames · 12 fps · 41.7s");
    expect(wrapper.text()).toContain("automatic sequence");
    expect(wrapper.emitted("update:frames")).toBeUndefined();
  });
});
