import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import AdvancedDrawer from "./AdvancedDrawer.vue";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import type { GenerateFormState, ModelInfoExtended } from "../../types";
import {
  hunyuan3dRecipe,
  sdxlRecipe,
} from "@studio/lib/generationProfile.testFixtures";

/** The Advanced sheet's exact-size field is the last place a canvas could be
 * typed into a canvasless request; it goes with Shape and Resolution. */

vi.mock("../../api", () => ({
  fetchModels: vi.fn(async () => []),
  fetchCatalogInstalled: vi.fn(async () => ({
    entries: [],
    page: 1,
    page_size: 0,
    total: 0,
  })),
}));

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({ ok: true, json: async () => [] }),
  );
});
afterEach(() => {
  vi.unstubAllGlobals();
  __testing__.resetForTest();
});

function model(
  name: string,
  family: string,
  recipe: ReturnType<typeof sdxlRecipe>,
): ModelInfoExtended {
  return {
    name,
    family,
    downloaded: true,
    default_steps: 20,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    generation_profile: {
      schema_version: 1,
      profile_id: family,
      profile_hash: family,
      default_recipe_id: "default",
      recipes: [recipe],
    },
  } as unknown as ModelInfoExtended;
}

function factory(
  family: string,
  overrides: Partial<GenerateFormState>,
  row: ModelInfoExtended,
) {
  const pinia = createPinia();
  setActivePinia(pinia);
  __testing__.resetForTest();
  const state = useGenerateForm().state.value;
  return mount(AdvancedDrawer, {
    props: {
      open: true,
      modelValue: { ...state, ...overrides },
      family,
      models: [row],
    },
    global: {
      plugins: [pinia],
      stubs: {
        LoraPicker: { template: "<div data-test='lora-picker-stub' />" },
        RouterLink: { template: "<a><slot /></a>" },
      },
    },
  });
}

describe("AdvancedDrawer canvasless recipes", () => {
  it("hides the exact-size field for a 3-D recipe", () => {
    const row = model(
      "hunyuan3d-mini-turbo:fp16",
      "hunyuan3d",
      hunyuan3dRecipe(),
    );
    const wrapper = factory(
      "hunyuan3d",
      {
        model: row.name,
        modelFamily: "hunyuan3d",
        width: 0,
        height: 0,
      },
      row,
    );
    expect(wrapper.find("[data-test='exact-width']").exists()).toBe(false);
    expect(wrapper.find("[data-test='exact-height']").exists()).toBe(false);
  });

  it("keeps the exact-size field for a raster recipe", () => {
    const row = model("sdxl:fp16", "sdxl", sdxlRecipe());
    const wrapper = factory(
      "sdxl",
      { model: row.name, modelFamily: "sdxl" },
      row,
    );
    expect(wrapper.find("[data-test='exact-width']").exists()).toBe(true);
  });
});
