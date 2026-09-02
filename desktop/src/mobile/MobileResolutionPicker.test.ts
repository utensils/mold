import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { defineComponent, reactive } from "vue";
import { describe, expect, it } from "vitest";
import { hunyuan3dRecipe } from "@studio/lib/generationProfile.testFixtures";
import MobileResolutionPicker from "./MobileResolutionPicker.vue";
import type { ModelEntry } from "../lib/api/types";

/** A Hunyuan3D checkpoint exactly as the host advertises one. */
const meshModel: ModelEntry = {
  name: "hunyuan3d-mini-turbo:fp16",
  family: "hunyuan3d",
  size_gb: 2,
  is_loaded: false,
  hf_repo: "tencent/Hunyuan3D-2mini",
  default_steps: 5,
  default_guidance: 5,
  default_width: 1024,
  default_height: 1024,
  description: "Mesh model",
  downloaded: true,
  source_image: "required",
  generation_profile: {
    schema_version: 1,
    profile_id: "hunyuan3d.mini",
    profile_hash: "hunyuan3d-mini-hash",
    default_recipe_id: "default",
    recipes: [hunyuan3dRecipe()],
  },
};

interface PickerState {
  width: number;
  height: number;
  family: string;
}

function mountPicker(
  width: number,
  height: number,
  family = "flux",
  sourceDimensions: { width: number; height: number } | null = null,
  model: ModelEntry | null = null,
  canvasIntent: "source" | "source-exact" | "model-default" | "manual" = "model-default",
) {
  const state = reactive<PickerState>({ width, height, family });
  const Harness = defineComponent({
    components: { MobileResolutionPicker },
    setup: () => ({ state, sourceDimensions, model, canvasIntent }),
    template: `
      <MobileResolutionPicker
        v-model:width="state.width"
        v-model:height="state.height"
        :family="state.family"
        :model="model"
        :source-dimensions="sourceDimensions"
        :canvas-intent="canvasIntent"
      />
    `,
  });
  return { wrapper: mount(Harness), state };
}

function tierSegments(wrapper: VueWrapper) {
  return wrapper.get("[data-test='mobile-resolution-tier']").findAll("button");
}

describe("MobileResolutionPicker", () => {
  it("uses the shared desktop shape picker without orientation tabs", async () => {
    const { wrapper, state } = mountPicker(1024, 1024, "flux");

    expect(wrapper.find("[data-test='mobile-resolution-summary']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-resolution-announcement']").text()).toBe(
      "Selected resolution: 1024 by 1024 pixels, 1:1, Square.",
    );
    expect(wrapper.find("[data-orientation]").exists()).toBe(false);
    expect(wrapper.get("[data-shape='1:1']").attributes("aria-checked")).toBe("true");
    const active = wrapper.get("[data-test='mobile-resolution-tier'] [aria-checked='true']");
    expect(active.get(".ms-seg__label").text()).toBe("1024×1024");
    expect(active.get(".ms-seg__sub").text()).toBe("1 MP");

    await wrapper.get("[data-shape='3:4']").trigger("click");
    expect(state).toMatchObject({ width: 768, height: 1024 });
    expect(wrapper.get("[data-test='mobile-resolution-announcement']").text()).toBe(
      "Selected resolution: 768 by 1024 pixels, 3:4, Portrait.",
    );
    expect(wrapper.get("[data-shape='3:4']").attributes("aria-checked")).toBe("true");

    await wrapper.get("[data-shape='9:16']").trigger("click");
    expect(state).toMatchObject({ width: 576, height: 1024 });
  });

  it("shows every Qwen Image aspect ratio on iPhone", async () => {
    const { wrapper, state } = mountPicker(1328, 1328, "qwen-image");

    expect(wrapper.find("[data-test='mobile-resolution-tier']").exists()).toBe(true);
    expect(wrapper.findAll(".ms-shape__btn").map((button) => button.text())).toEqual([
      "1:1",
      "4:3",
      "3:4",
      "3:2",
      "2:3",
      "16:9",
      "9:16",
    ]);
    await wrapper.get("[data-shape='9:16']").trigger("click");
    expect(state).toMatchObject({ width: 928, height: 1664 });
    await wrapper.get("[data-shape='16:9']").trigger("click");
    expect(state).toMatchObject({ width: 1664, height: 928 });
  });

  it("projects the selected tier's exact pixel dimensions under the control", async () => {
    const { wrapper, state } = mountPicker(1024, 1024, "flux");

    expect(wrapper.get("[data-test='mobile-resolution-tier-dims']").text()).toBe("1024 × 1024 px");

    await tierSegments(wrapper)[0]?.trigger("click");
    expect(state).toMatchObject({ width: 768, height: 768 });
    expect(wrapper.get("[data-test='mobile-resolution-tier-dims']").text()).toBe("768 × 768 px");
  });

  it("labels a source-matched custom canvas and restores it after a manual override", async () => {
    const sourceDimensions = { width: 896, height: 1152 };
    const { wrapper, state } = mountPicker(896, 1152, "qwen-image-edit", sourceDimensions);

    expect(wrapper.get("[data-test='mobile-source-resolution-status']").text()).toContain(
      "896×1152 · Matches source",
    );
    expect(wrapper.find("[data-test='mobile-match-source-resolution']").exists()).toBe(false);

    state.width = 1024;
    state.height = 1024;
    await flushPromises();
    const status = wrapper.get("[data-test='mobile-source-resolution-status']").text();
    expect(status).toContain("Manual");
    expect(status).toContain("1024×1024");
    expect(status).not.toContain("Matches source");
    await wrapper.get("[data-test='mobile-match-source-resolution']").trigger("click");
    expect(state).toMatchObject(sourceDimensions);
  });

  it("lists every authored size in the family as exact pixels", () => {
    const single = tierSegments(mountPicker(1024, 1024, "flux").wrapper);
    expect(single).toHaveLength(2);
    expect(single.map((segment) => segment.get(".ms-seg__label").text())).toEqual([
      "768×768",
      "1024×1024",
    ]);
    expect(single[1]?.attributes("aria-checked")).toBe("true");

    // Every widescreen bucket LTX-2 authors — 16:9, 19:11, 30:17 — is one
    // canonical family whose ladder shows the real pixels (#1166).
    const wide = tierSegments(mountPicker(1024, 576, "ltx2").wrapper);
    expect(wide.map((segment) => segment.get(".ms-seg__label").text())).toEqual([
      "1024×576",
      "1216×704",
      "1920×1088",
    ]);
    expect(wide[0]?.get(".ms-seg__sub").text()).toBe("0.6 MP");
    expect(wide[0]?.attributes("aria-checked")).toBe("true");
  });

  it("renders shared proportional shape swatches", () => {
    const { wrapper } = mountPicker(704, 1216, "ltx2");
    const choices = wrapper.findAll(".ms-shape__btn");
    expect(choices.map((choice) => choice.text())).toEqual(["1:1", "3:2", "2:3", "16:9", "9:16"]);
    const frames = wrapper.findAll(".ms-shape__swatch");
    expect(frames).toHaveLength(5);
    expect(frames[0]?.attributes("style")).toContain("width: 24px");
    expect(frames[3]?.attributes("style")).toContain("width: 27px");
    expect(frames[4]?.attributes("style")).toContain("height: 27px");
  });

  it("falls back to iPhone-friendly custom fields and snaps values to 16", async () => {
    const { wrapper, state } = mountPicker(1000, 777);

    expect(wrapper.find("[data-test='mobile-resolution-tier']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-resolution-tier-dims']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-resolution-custom']").exists()).toBe(true);
    expect(wrapper.get("[data-test='mobile-resolution-announcement']").text()).toBe(
      "Selected resolution: 1000 by 777 pixels, 5:4, Landscape.",
    );

    const width = wrapper.get("input[aria-label='Custom width']");
    const height = wrapper.get("input[aria-label='Custom height']");
    expect(width.attributes()).toMatchObject({ inputmode: "numeric", min: "64", step: "16" });
    expect(height.attributes()).toMatchObject({ inputmode: "numeric", min: "64", step: "16" });

    await width.setValue("1001");
    await width.trigger("change");
    await height.setValue("20");
    await height.trigger("change");
    expect(state).toMatchObject({ width: 1008, height: 64 });

    await wrapper.get("[aria-label='Swap width and height']").trigger("click");
    expect(state).toMatchObject({ width: 64, height: 1008 });
  });

  it("uses the selected model's /32 grid for custom dimensions", async () => {
    const { wrapper, state } = mountPicker(1000, 776, "ltx2", null, {
      name: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      dimension_alignment: 32,
    } as ModelEntry);
    const width = wrapper.get("input[aria-label='Custom width']");
    const height = wrapper.get("input[aria-label='Custom height']");
    expect(width.attributes("step")).toBe("32");
    expect(height.attributes("step")).toBe("32");
    await width.trigger("change");
    await height.trigger("change");
    expect(state).toMatchObject({ width: 992, height: 768 });
  });

  it("lets a preset size reveal manual fields without changing dimensions", async () => {
    const { wrapper, state } = mountPicker(1024, 1024);

    expect(wrapper.find("[data-test='mobile-resolution-custom']").exists()).toBe(false);
    const toggle = wrapper.get("[data-test='mobile-resolution-custom-toggle']");
    expect(toggle.attributes("aria-expanded")).toBe("false");

    await toggle.trigger("click");
    expect(wrapper.find("[data-test='mobile-resolution-custom']").exists()).toBe(true);
    expect(state).toMatchObject({ width: 1024, height: 1024 });
    expect(toggle.text()).toBe("Hide custom size");
  });

  it("highlights the closest aspect chip as approximate for a custom size", () => {
    const { wrapper } = mountPicker(1000, 600, "flux", null, {
      name: "flux:profiled",
      family: "flux",
      recommended_dimensions: [{ width: 1024, height: 576 }],
    } as ModelEntry);
    const approximate = wrapper.get("[data-approximate='true']");
    expect(approximate.text()).toContain("≈");
    expect(approximate.attributes("aria-label")).toContain("closest match");
  });

  it("keeps exact presets unmarked", () => {
    const { wrapper } = mountPicker(1024, 1024, "flux");
    expect(wrapper.find("[data-approximate='true']").exists()).toBe(false);
    expect(wrapper.get("[data-shape='1:1']").text()).not.toContain("≈");
  });

  it("defaults a source-sized preset to the model aspect but keeps Source selectable", async () => {
    const { wrapper } = mountPicker(1024, 1024, "flux", {
      width: 1024,
      height: 1024,
    });

    expect(wrapper.get("[data-shape='1:1']").attributes("aria-checked")).toBe("true");
    await wrapper.get("[data-shape='source']").trigger("click");
    expect(wrapper.getComponent(MobileResolutionPicker).emitted("canvas-intent")?.at(-1)).toEqual([
      "source",
    ]);

    wrapper.unmount();
    // The parent owns the intent; once it flows back the Source chip wins.
    const remounted = mountPicker(
      1024,
      1024,
      "flux",
      { width: 1024, height: 1024 },
      null,
      "source",
    ).wrapper;
    expect(remounted.get("[data-shape='source']").attributes("aria-checked")).toBe("true");
  });

  it("applies a square Source canvas for Wan instead of its landscape default", async () => {
    const wan = {
      name: "wan22-ti2v-5b:fp16",
      family: "wan",
      default_width: 832,
      default_height: 480,
      recommended_dimensions: [
        { width: 832, height: 480 },
        { width: 480, height: 832 },
        { width: 1280, height: 720 },
        { width: 720, height: 1280 },
      ],
      max_pixels: 1_800_000,
      dimension_alignment: 16,
    } as ModelEntry;
    const { wrapper, state } = mountPicker(
      832,
      480,
      "wan",
      { width: 1024, height: 1024 },
      wan,
      "model-default",
    );

    await wrapper.get("[data-shape='source']").trigger("click");

    expect(state).toMatchObject({ width: 1024, height: 1024 });
    expect(wrapper.getComponent(MobileResolutionPicker).emitted("canvas-intent")?.at(-1)).toEqual([
      "source",
    ]);
  });

  it("advises on custom sizes above the 1.8 MP guideline without blocking", async () => {
    // The server is the authority: an oversized custom size submits anyway
    // and its refusal (if any) comes back as the job's own error.
    const { wrapper, state } = mountPicker(2000, 2000);
    const picker = wrapper.getComponent(MobileResolutionPicker);

    expect(wrapper.find("[data-test='mobile-resolution-error']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-resolution-warning']").text()).toContain("1.8 MP");
    expect(picker.emitted("validity-change")?.at(-1)).toEqual([true]);

    state.width = 1328;
    state.height = 1328;
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-resolution-warning']").exists()).toBe(false);
    expect(picker.emitted("validity-change")?.at(-1)).toEqual([true]);
  });

  it("exposes portrait and landscape shapes together from the shared contract", async () => {
    const { wrapper, state } = mountPicker(1024, 576, "ltx2");

    expect(wrapper.findAll(".ms-shape__btn").map((button) => button.text())).toEqual([
      "1:1",
      "3:2",
      "2:3",
      "16:9",
      "9:16",
    ]);
    expect(wrapper.get("[data-shape='16:9']").attributes("aria-checked")).toBe("true");
    await wrapper.get("[data-shape='9:16']").trigger("click");
    await flushPromises();
    expect(state).toMatchObject({ width: 576, height: 1024 });
  });

  it("exposes controlled width and height update events", async () => {
    const wrapper = mount(MobileResolutionPicker, {
      props: { family: "sdxl", width: 1024, height: 1024 },
    });

    // SDXL authors 896×1152, which is 4:5 rather than a true 3:4.
    await wrapper.get("[data-shape='4:5']").trigger("click");
    expect(wrapper.emitted("update:width")).toEqual([[896]]);
    expect(wrapper.emitted("update:height")).toEqual([[1152]]);
  });

  /**
   * A canvasless recipe renders a mesh, not pixels: there is no shape, no
   * ladder and no custom size to offer, and the zero canvas it carries is
   * correct rather than malformed input. Blocking on `width < 1` here would
   * disable Develop for every 3-D print.
   */
  it("renders nothing and stays valid on a canvasless recipe", () => {
    const wrapper = mount(MobileResolutionPicker, {
      props: {
        family: meshModel.family,
        model: meshModel,
        width: 0,
        height: 0,
      },
    });

    expect(wrapper.find(".mobile-resolution-picker").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-resolution-error']").exists()).toBe(false);
    expect(wrapper.emitted("validity-change")).toEqual([[true]]);
  });
});
