/**
 * The Create inspector's Transparent background row (Qwen Image 2.1). It is
 * rendered from the resolved recipe's own `capabilities.transparency` block —
 * never a family name — and turning it on moves a JPEG choice to the recipe's
 * first alpha format, so the form never holds a pair admission refuses.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import InspectorPanel from "./InspectorPanel.vue";
import { qwenImage21Recipe, sdxlRecipe } from "@studio/lib/generationProfile.testFixtures";
import type { GenerationRecipeProfile } from "@studio/lib/generationProfile";
import {
  applyModelDefaults,
  buildRequest,
  newGenerateForm,
  type GenerateForm,
} from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";
import { useModelStore } from "../../stores/models";
import { useHostModelsStore } from "../../stores/hostModels";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("../../lib/ipc", () => ({ ipc: {}, inTauri: () => false }));
vi.mock("@studio/api/galleryOrganization", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  listCollections: vi.fn(() => Promise.resolve([])),
  listTags: vi.fn(() => Promise.resolve([])),
}));

function modelWith(name: string, family: string, recipe: GenerationRecipeProfile): ModelEntry {
  return {
    name,
    family,
    downloaded: true,
    default_steps: recipe.defaults.steps,
    default_guidance: recipe.defaults.guidance,
    default_width: 1024,
    default_height: 1024,
    generation_profile: {
      schema_version: 1,
      profile_id: `${family}.${name}`,
      profile_hash: "hash",
      default_recipe_id: recipe.id,
      recipes: [recipe],
    },
  } as unknown as ModelEntry;
}

function mountFor(model: ModelEntry) {
  useModelStore().all = [model];
  useHostModelsStore().byHost.local = { entries: [model], fetchedAt: Date.now(), error: null };
  const form = reactive(newGenerateForm()) as GenerateForm;
  form.model = model.name;
  form.family = model.family;
  applyModelDefaults(form, model);
  const wrapper = mount(InspectorPanel, { props: { form } });
  return { form, wrapper };
}

describe("InspectorPanel transparent background", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
  });

  it("renders only where the recipe advertises the toggle", () => {
    const qwen = mountFor(modelWith("qwen-image-2.1:bf16", "qwen-image21", qwenImage21Recipe()));
    expect(qwen.wrapper.find("[data-test='transparent-background']").exists()).toBe(true);
    const sdxl = mountFor(modelWith("sdxl-base:fp16", "sdxl", sdxlRecipe()));
    expect(sdxl.wrapper.find("[data-test='transparent-background']").exists()).toBe(false);
  });

  it("turning it on sends the field and moves JPEG to PNG", async () => {
    const { form, wrapper } = mountFor(
      modelWith("qwen-image-2.1:bf16", "qwen-image21", qwenImage21Recipe()),
    );
    form.prompt = "a paper lantern";
    form.outputFormat = "jpeg";
    await wrapper.get("[data-test='transparent-background']").trigger("click");
    expect(form.transparentBackground).toBe(true);
    expect(form.outputFormat).toBe("png");
    const req = buildRequest(form);
    expect(req.transparent_background).toBe(true);
    expect(req.output_format).toBe("png");
  });
});
