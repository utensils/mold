/**
 * A canvasless 3-D recipe has no canvas to fit the source onto, and
 * Hunyuan3D reads neither a denoise strength nor a repaint mask — the source
 * well must not offer three controls the request cannot carry. Every answer
 * comes from the selected model's advertised recipe, never a family
 * allowlist: the pre-profile rules say hunyuan3d supports both.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises, mount } from "@vue/test-utils";
import { reactive } from "vue";
import SourceImageWell from "./SourceImageWell.vue";
import { hunyuan3dRecipe, sdxlRecipe } from "@studio/lib/generationProfile.testFixtures";
import type { GenerationRecipeProfile } from "@studio/lib/generationProfile";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";

vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetchTo: vi.fn(),
  currentTarget: vi.fn(() => ({ baseUrl: "http://127.0.0.1:49152", apiKey: null })),
}));
const { fetchCatalogInstalled } = vi.hoisted(() => ({
  fetchCatalogInstalled: vi.fn(() =>
    Promise.resolve({ entries: [], page: 1, page_size: 0, total: 0 }),
  ),
}));
vi.mock("../../lib/api/catalog", () => ({ fetchCatalogInstalled }));

function modelWith(name: string, family: string, recipe: GenerationRecipeProfile): ModelEntry {
  return {
    name,
    family,
    downloaded: true,
    ...(family === "hunyuan3d" ? { source_image: "required" } : {}),
    generation_profile: {
      schema_version: 1,
      profile_id: `${family}.${name}`,
      profile_hash: "hash",
      default_recipe_id: recipe.id,
      recipes: [recipe],
    },
  } as unknown as ModelEntry;
}

function formFor(family: string, model: string): GenerateForm {
  const form = reactive({ ...newGenerateForm(), family, model }) as GenerateForm;
  form.sourceImage = "U1JD";
  form.sourceImageName = "armchair.png";
  return form;
}

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => (document.body.innerHTML = ""));

describe("SourceImageWell — canvasless 3-D recipes", () => {
  it("offers no source fit, strength or mask for an attached mesh source", async () => {
    const selectedModel = modelWith("hunyuan3d-mini-turbo:fp16", "hunyuan3d", hunyuan3dRecipe());
    const wrapper = mount(SourceImageWell, {
      props: { form: formFor("hunyuan3d", selectedModel.name), selectedModel },
      attachTo: document.body,
    });
    await flushPromises();
    expect(wrapper.find("[data-test='source-media-wells']").exists()).toBe(true);
    expect(wrapper.find("[data-test='source-fit-policy']").exists()).toBe(false);
    expect(wrapper.vm.maskAvailable).toBe(false);
    expect(wrapper.text()).not.toContain("Prompt strength");
  });

  it("keeps source fit, strength and the mask for a raster recipe", async () => {
    const selectedModel = modelWith("sdxl-base:fp16", "sdxl", sdxlRecipe());
    const wrapper = mount(SourceImageWell, {
      props: { form: formFor("sdxl", selectedModel.name), selectedModel },
      attachTo: document.body,
    });
    await flushPromises();
    expect(wrapper.find("[data-test='source-fit-policy']").exists()).toBe(true);
    expect(wrapper.vm.maskAvailable).toBe(true);
  });
});
