/**
 * Advanced's Output & seed section offers an exact pixel size. A canvasless
 * recipe (a 3-D mesh) renders at no pixel size at all, so typing one would
 * steer a canvas the request ignores — and typing over the recipe's zero
 * canvas is exactly how a 0×0 request becomes a spurious 422.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import AdvancedSettings from "./AdvancedSettings.vue";
import { hunyuan3dRecipe, sdxlRecipe } from "@studio/lib/generationProfile.testFixtures";
import type { GenerationRecipeProfile } from "@studio/lib/generationProfile";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";

vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("../../lib/ipc", () => ({
  ipc: {
    localGalleryList: vi.fn(() => Promise.resolve({ images: [], target: null })),
    localGalleryDelete: vi.fn(),
  },
  inTauri: () => false,
}));

function modelWith(name: string, family: string, recipe: GenerationRecipeProfile): ModelEntry {
  return {
    name,
    family,
    downloaded: true,
    generation_profile: {
      schema_version: 1,
      profile_id: `${family}.${name}`,
      profile_hash: "hash",
      default_recipe_id: recipe.id,
      recipes: [recipe],
    },
  } as unknown as ModelEntry;
}

function mountFor(selectedModel: ModelEntry) {
  const form = reactive({
    ...newGenerateForm(),
    family: selectedModel.family,
    model: selectedModel.name,
  }) as GenerateForm;
  return mount(AdvancedSettings, { props: { form, selectedModel }, attachTo: document.body });
}

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => (document.body.innerHTML = ""));

describe("AdvancedSettings — canvasless recipes", () => {
  it("hides the exact-size field for a 3-D recipe", async () => {
    const wrapper = mountFor(
      modelWith("hunyuan3d-mini-turbo:fp16", "hunyuan3d", hunyuan3dRecipe()),
    );
    await flushPromises();
    expect(wrapper.find("[data-test='advanced-exact-size']").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("Exact size");
  });

  it("keeps the exact-size field for a raster recipe", async () => {
    const wrapper = mountFor(modelWith("sdxl-base:fp16", "sdxl", sdxlRecipe()));
    await flushPromises();
    expect(wrapper.find("[data-test='advanced-exact-size']").exists()).toBe(true);
  });
});
