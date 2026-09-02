import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import SourceMediaPanel from "./SourceMediaPanel.vue";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import type { GenerateFormState, ModelInfoExtended } from "../../types";
import {
  hunyuan3dRecipe,
  sdxlRecipe,
} from "@studio/lib/generationProfile.testFixtures";

/**
 * A canvasless recipe has no canvas to fit the source onto, and Hunyuan3D
 * reads neither a denoise strength nor a repaint mask — the source card must
 * not offer three controls the request cannot carry.
 */

function meshModel(): ModelInfoExtended {
  return {
    name: "hunyuan3d-mini-turbo:fp16",
    family: "hunyuan3d",
    size_gb: 3,
    is_loaded: false,
    last_used: null,
    hf_repo: "tencent/Hunyuan3D-2mini",
    downloaded: true,
    default_steps: 5,
    default_guidance: 5,
    default_width: 1024,
    default_height: 1024,
    description: "",
    source_image: "required",
    generation_profile: {
      schema_version: 1,
      profile_id: "hunyuan3d",
      profile_hash: "h3d",
      default_recipe_id: "default",
      recipes: [hunyuan3dRecipe()],
    },
  } as ModelInfoExtended;
}

function rasterModel(): ModelInfoExtended {
  return {
    name: "sdxl:fp16",
    family: "sdxl",
    size_gb: 6,
    is_loaded: false,
    last_used: null,
    hf_repo: "stabilityai/sdxl",
    downloaded: true,
    default_steps: 25,
    default_guidance: 7,
    default_width: 1024,
    default_height: 1024,
    description: "",
    generation_profile: {
      schema_version: 1,
      profile_id: "sdxl",
      profile_hash: "sdxl",
      default_recipe_id: "default",
      recipes: [sdxlRecipe()],
    },
  } as ModelInfoExtended;
}

function baseForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  __testing__.resetForTest();
  const state = useGenerateForm().state.value;
  return { ...state, ...overrides };
}

const attachment = {
  kind: "upload" as const,
  filename: "chair.png",
  base64: "AAA",
};

beforeEach(() => localStorage.clear());
afterEach(() => __testing__.resetForTest());

describe("SourceMediaPanel — canvasless 3-D recipes", () => {
  it("offers no fit, strength or mask for an attached mesh source", () => {
    const wrapper = mount(SourceMediaPanel, {
      props: {
        modelValue: baseForm({
          model: "hunyuan3d-mini-turbo:fp16",
          modelFamily: "hunyuan3d",
          width: 0,
          height: 0,
          imageAttachments: [attachment],
        }),
        family: "hunyuan3d",
        models: [meshModel()],
      },
    });
    expect(wrapper.find("[data-test='source-media-wells']").exists()).toBe(
      true,
    );
    expect(wrapper.text()).not.toContain("Fit to canvas");
    expect(wrapper.find("[data-test='source-mask']").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("Prompt strength");
  });

  it("keeps the fit control for a raster recipe", () => {
    const wrapper = mount(SourceMediaPanel, {
      props: {
        modelValue: baseForm({
          model: "sdxl:fp16",
          modelFamily: "sdxl",
          imageAttachments: [attachment],
        }),
        family: "sdxl",
        models: [rasterModel()],
      },
    });
    expect(wrapper.text()).toContain("Fit to canvas");
    expect(wrapper.find("[data-test='source-mask']").exists()).toBe(true);
  });
});
