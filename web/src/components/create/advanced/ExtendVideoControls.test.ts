import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ExtendVideoControls from "./ExtendVideoControls.vue";
import {
  useGenerateForm,
  __testing__,
} from "../../../composables/useGenerateForm";
import type { GenerateFormState } from "../../../types";

function baseForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  __testing__.resetForTest();
  const state = useGenerateForm().state.value;
  return { ...state, ...overrides };
}

function factory(
  overrides: Partial<GenerateFormState> = {},
  extra: { family?: string; defaultOverlapFrames?: number } = {},
) {
  return mount(ExtendVideoControls, {
    props: { modelValue: baseForm(overrides), ...extra },
  });
}

function lastPatch(wrapper: ReturnType<typeof factory>): GenerateFormState {
  const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
    GenerateFormState,
  ];
  return next;
}

describe("ExtendVideoControls", () => {
  it("only shows the overlap control once a video is attached", async () => {
    const wrapper = factory({}, { family: "ltx2" });
    expect(wrapper.find("[data-test='extend-overlap']").exists()).toBe(false);

    await wrapper.get("[data-test='extend-path']").setValue("/srv/clip.mp4");
    await wrapper.setProps({ modelValue: lastPatch(wrapper) });
    expect(wrapper.find("[data-test='extend-overlap']").exists()).toBe(true);
  });

  it("reports how many frames the continuation actually appends", async () => {
    const wrapper = factory({ frames: 97 }, { family: "ltx2" });
    await wrapper.get("[data-test='extend-path']").setValue("/srv/clip.mp4");
    await wrapper.setProps({ modelValue: lastPatch(wrapper) });

    // 97 rendered frames minus the 17-frame overlap that re-renders the tail.
    expect(wrapper.get("[data-test='extend-summary']").text()).toContain(
      "80 new frames",
    );
  });

  it("explains a conflicting source image before the request is sent", async () => {
    const wrapper = factory(
      {
        frames: 97,
        imageAttachments: [
          { base64: "AAAA", filename: "a.png", mime: "image/png" },
        ] as GenerateFormState["imageAttachments"],
      },
      { family: "ltx2" },
    );
    await wrapper.get("[data-test='extend-path']").setValue("/srv/clip.mp4");
    await wrapper.setProps({ modelValue: lastPatch(wrapper) });

    expect(wrapper.get("[data-test='extend-error']").text()).toContain(
      "source image",
    );
  });

  // Clearing must also drop the overlap: a value valid for the old clip can be
  // invalid for the next one, and a stale number would silently ride along.
  it("resets the overlap when the attached video is removed", async () => {
    const wrapper = factory({ frames: 97 }, { family: "ltx2" });
    await wrapper.setProps({
      modelValue: {
        ...baseForm({ frames: 97 }),
        extendVideo: {
          base64: "AAAA",
          filename: "clip.mp4",
          mime: "video/mp4",
        } as GenerateFormState["extendVideo"],
        extendOverlapFrames: 33,
      },
    });

    await wrapper.get("[data-test='extend-clear']").trigger("click");
    const next = lastPatch(wrapper);
    expect(next.extendVideo).toBeNull();
    expect(next.extendOverlapFrames).toBeNull();
  });

  /**
   * Wan is seeded with the source's final frame and carries no latent motion
   * tail (#783), so its engine accepts exactly one overlap. The shared 8k+1
   * ladder — and the server's family-wide 17-frame default — would both have
   * produced a rejected request.
   */
  it("offers wan its single carried frame and explains any other value", async () => {
    const wrapper = factory(
      { frames: 53, extendVideoPath: "/srv/clip.mp4", extendOverlapFrames: 1 },
      { family: "wan", defaultOverlapFrames: 1 },
    );
    expect(
      wrapper
        .get("[data-test='extend-overlap']")
        .findAll("option")
        .map((option) => option.text()),
    ).toEqual(["1"]);
    expect(wrapper.get("[data-test='extend-summary']").text()).toContain(
      "52 new frames",
    );

    await wrapper.setProps({
      modelValue: {
        ...baseForm({ frames: 53, extendVideoPath: "/srv/clip.mp4" }),
        extendOverlapFrames: 17,
      },
    });
    expect(wrapper.get("[data-test='extend-error']").text()).toContain(
      "exactly 1 frame",
    );
  });
});
