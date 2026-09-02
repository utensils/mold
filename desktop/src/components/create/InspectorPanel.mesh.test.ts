/**
 * The Create inspector on a canvasless 3-D recipe: Shape and Resolution have
 * nothing to bind to and disappear, and a Mesh group built entirely from the
 * recipe's own advertised `capabilities.mesh` block takes their place after
 * Detail. Every bound is the server's — widening the octree ladder or the
 * face range on a host widens this group with no client release.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import InspectorPanel from "./InspectorPanel.vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import { hunyuan3dRecipe, sdxlRecipe } from "@studio/lib/generationProfile.testFixtures";
import type { GenerationRecipeProfile } from "@studio/lib/generationProfile";
import { applyModelDefaults, newGenerateForm, type GenerateForm } from "../../lib/generateForm";
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

const meshModel = () => modelWith("hunyuan3d-mini-turbo:fp16", "hunyuan3d", hunyuan3dRecipe());
const rasterModel = () => modelWith("sdxl-base:fp16", "sdxl", sdxlRecipe());

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

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => (document.body.innerHTML = ""));

describe("InspectorPanel — canvasless recipes", () => {
  it("hides Shape and Resolution when the recipe renders no canvas", async () => {
    const { wrapper } = mountFor(meshModel());
    await flushPromises();
    expect(wrapper.findComponent(ShapePicker).exists()).toBe(false);
    expect(wrapper.findComponent(ResolutionSelector).exists()).toBe(false);
  });

  it("keeps Shape and Resolution for a raster recipe", async () => {
    const { wrapper } = mountFor(rasterModel());
    await flushPromises();
    expect(wrapper.findComponent(ShapePicker).exists()).toBe(true);
    expect(wrapper.findComponent(ResolutionSelector).exists()).toBe(true);
  });
});

describe("InspectorPanel — Mesh group", () => {
  it("renders nothing for a recipe with no mesh block", async () => {
    const { wrapper } = mountFor(rasterModel());
    await flushPromises();
    expect(wrapper.find("[data-test='mesh-controls']").exists()).toBe(false);
  });

  it("builds the octree ladder from the recipe and lights the advertised default", async () => {
    const { wrapper } = mountFor(meshModel());
    await flushPromises();
    const octree = wrapper
      .findAllComponents({ name: "SegmentedControl" })
      .find((row) => row.attributes("data-test") === "mesh-octree")!;
    expect(octree.props("options")).toEqual([
      { value: 128, label: "128" },
      { value: 192, label: "192" },
      { value: 256, label: "256" },
      { value: 320, label: "320" },
      { value: 384, label: "384" },
    ]);
    // Untouched means "use the profile default", and the default is lit.
    expect(octree.props("modelValue")).toBe(256);
  });

  it("writes the chosen octree resolution onto the form", async () => {
    const { form, wrapper } = mountFor(meshModel());
    await flushPromises();
    const octree = wrapper
      .findAllComponents({ name: "SegmentedControl" })
      .find((row) => row.attributes("data-test") === "mesh-octree")!;
    octree.vm.$emit("update:modelValue", 384);
    await flushPromises();
    expect(form.mesh.octreeResolution).toBe(384);
  });

  it("takes the iso threshold's bounds and default from the FloatControl", async () => {
    const { form, wrapper } = mountFor(meshModel());
    await flushPromises();
    const threshold = wrapper
      .findAllComponents(SliderRow)
      .find((row) => row.props("label") === "Iso threshold")!;
    expect(threshold.props()).toMatchObject({
      min: 0,
      max: 1,
      step: 0.01,
      modelValue: 0.6,
      disabled: false,
    });
    expect(wrapper.find("[data-test='mesh-threshold-note']").exists()).toBe(false);
    threshold.vm.$emit("update:modelValue", 0.42);
    await flushPromises();
    expect(form.mesh.threshold).toBe(0.42);
  });

  it("disables a fixed threshold and shows the profile's own note", async () => {
    const recipe = hunyuan3dRecipe();
    recipe.capabilities.mesh!.threshold = {
      default: 0.5,
      min: 0.5,
      max: 0.5,
      step: 0.01,
      mode: "fixed",
      note: "This build pins the iso surface.",
    };
    const { wrapper } = mountFor(modelWith("hunyuan3d-fixed:fp16", "hunyuan3d", recipe));
    await flushPromises();
    const threshold = wrapper
      .findAllComponents(SliderRow)
      .find((row) => row.props("label") === "Iso threshold")!;
    expect(threshold.props("disabled")).toBe(true);
    expect(wrapper.get("[data-test='mesh-threshold-note']").text()).toBe(
      "This build pins the iso surface.",
    );
  });

  it("bounds Target faces by the recipe and treats blank as the raw surface", async () => {
    const { form, wrapper } = mountFor(meshModel());
    await flushPromises();
    const faces = wrapper.get("[data-test='mesh-target-faces']");
    expect(faces.attributes("min")).toBe("100");
    expect(faces.attributes("max")).toBe("2000000");
    expect((faces.element as HTMLInputElement).value).toBe("");

    await faces.setValue("50000");
    expect(form.mesh.targetFaces).toBe(50_000);
    await faces.setValue("");
    expect(form.mesh.targetFaces).toBeNull();
  });
});
