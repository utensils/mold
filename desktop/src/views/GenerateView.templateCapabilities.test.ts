/**
 * Loading a template re-reads the selected model's ADVERTISED capabilities.
 * A template is a set of parameters, not a capability snapshot: one saved
 * before the mesh controls existed (or on another host) carries no
 * `recipeCapabilities`, so applying it wholesale over a Hunyuan3D form left
 * the mesh snapshot in place and pinned `glb` onto the SDXL request.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises, mount } from "@vue/test-utils";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";

installMemoryLocalStorage();

import GenerateView from "./GenerateView.vue";
import InspectorPanel from "../components/create/InspectorPanel.vue";
import { useGenerateFormStore } from "../stores/generateForm";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { applyModelDefaults, buildRequest, newGenerateForm } from "../lib/generateForm";
import type { GenerationTemplate } from "../lib/generationTemplates";
import { hunyuan3dRecipe, sdxlRecipe } from "@studio/lib/generationProfile.testFixtures";
import type { GenerationRecipeProfile } from "@studio/lib/generationProfile";
import type { ModelEntry } from "../lib/api/types";

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
}));
vi.mock("../lib/ipc", () => ({ ipc: {} }));

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

const mesh = modelWith("hunyuan3d-mini-turbo:fp16", "hunyuan3d", hunyuan3dRecipe());
const sdxl = modelWith("sdxl-base:fp16", "sdxl", sdxlRecipe());

/** A template saved before the form carried a capability snapshot. */
function legacyTemplate(model: string, family: string): GenerationTemplate {
  const saved = newGenerateForm() as unknown as Record<string, unknown>;
  saved.model = model;
  saved.family = family;
  saved.prompt = "a river at dawn";
  saved.width = 1024;
  saved.height = 1024;
  delete saved.recipeCapabilities;
  delete saved.mesh;
  return {
    id: "legacy",
    name: "River preset",
    createdAt: 1,
    updatedAt: 1,
    form: saved as unknown as GenerationTemplate["form"],
    mediaReferences: [],
  };
}

async function loadIntoMeshForm(template: GenerationTemplate) {
  const wrapper = mount(GenerateView, { shallow: true, attachTo: document.body });
  await flushPromises();
  useModelStore().all = [mesh, sdxl];
  await flushPromises();

  const form = useGenerateFormStore().form;
  form.model = mesh.name;
  form.family = mesh.family;
  applyModelDefaults(form, mesh);
  form.mesh.octreeResolution = 192;
  expect(form.recipeCapabilities?.canvasless).toBe(true);

  // Starting points are a tab in the inspector now, not a floating popover:
  // the panel loads the template and the view answers on `load-template`.
  wrapper.findComponent(InspectorPanel).vm.$emit("load-template", template);
  await flushPromises();
  return form;
}

describe("GenerateView — loading a template refreshes the recipe capabilities", () => {
  beforeEach(() => {
    localStorage.clear();
    setActivePinia(createPinia());
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
    conn.status = "ready";
    useHostsStore().initialized = true;
    useModelStore().all = [mesh, sdxl];
  });
  afterEach(() => {
    document.body.innerHTML = "";
    localStorage.clear();
  });

  it("reads the installed model's recipe instead of keeping the mesh snapshot", async () => {
    const form = await loadIntoMeshForm(legacyTemplate(sdxl.name, sdxl.family));
    expect(form.model).toBe(sdxl.name);
    expect(form.recipeCapabilities?.canvasless).toBe(false);
    expect(form.recipeCapabilities?.mesh).toBeNull();
    expect(form.mesh).toEqual({ octreeResolution: null, threshold: null, targetFaces: null });
    expect(form.width).toBe(1024);
    const request = buildRequest(form);
    expect(request.output_format).toBe("png");
    expect(request).not.toHaveProperty("mesh");
  });

  it("drops the snapshot when the template's model is not installed", async () => {
    const form = await loadIntoMeshForm(legacyTemplate("flux-dev:q8", "flux"));
    expect(form.model).toBe("flux-dev:q8");
    expect(form.recipeCapabilities).toBeNull();
    expect(buildRequest(form).output_format).not.toBe("glb");
  });
});
