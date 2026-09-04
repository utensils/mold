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
import { useHostsStore } from "../../stores/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";

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

function readyHost(id: string, label: string): void {
  const hosts = useHostsStore();
  hosts.extras.push({
    id,
    label,
    url: `http://${id}.test:7680`,
    apiKey: null,
    status: "ready",
    error: null,
    instanceId: `instance-${id}`,
  } as never);
  hosts.telemetry[id] = { instanceId: `instance-${id}` } as never;
}

/**
 * The mesh model is installed on ONE machine, and Create is aimed at another
 * that would have to download it first. The Model field still names it (the
 * picker resolves it across every machine and the catalog), so the settings
 * beneath must answer for the same checkpoint.
 */
function mountForTargetWithoutModel(model: ModelEntry) {
  useModelStore().all = [];
  readyHost("halcyon", "halcyon");
  readyHost("plato", "plato");
  const hostModels = useHostModelsStore();
  hostModels.byHost.halcyon = { entries: [model], fetchedAt: Date.now(), error: null };
  hostModels.byHost.plato = { entries: [], fetchedAt: Date.now(), error: null };
  useAppPrefsStore().settings = { generateTargetHost: "plato" } as never;

  const form = reactive(newGenerateForm()) as GenerateForm;
  form.model = model.name;
  form.family = model.family;
  applyModelDefaults(form, model);
  // The strength slider and the mask well both render only once a source is
  // attached, which is exactly the state a Hunyuan3D print is generated in.
  form.sourceImage = "c291cmNl";
  form.sourceImageName = "armchair-cutout.png";
  const wrapper = mount(InspectorPanel, { props: { form } });
  return { form, wrapper };
}

/** The same two machines, with Create aimed at the one that HAS the model. */
function mountForTargetWithModel(model: ModelEntry) {
  useModelStore().all = [];
  readyHost("halcyon", "halcyon");
  readyHost("plato", "plato");
  const hostModels = useHostModelsStore();
  hostModels.byHost.halcyon = { entries: [model], fetchedAt: Date.now(), error: null };
  hostModels.byHost.plato = { entries: [], fetchedAt: Date.now(), error: null };
  useAppPrefsStore().settings = { generateTargetHost: "halcyon" } as never;

  const form = reactive(newGenerateForm()) as GenerateForm;
  form.model = model.name;
  form.family = model.family;
  applyModelDefaults(form, model);
  form.sourceImage = "c291cmNl";
  form.sourceImageName = "armchair-cutout.png";
  const wrapper = mount(InspectorPanel, { props: { form } });
  return { form, wrapper };
}

/** What the panel offers for the print, as the four leaking controls show it. */
async function panelContract(wrapper: ReturnType<typeof mount>) {
  await wrapper.get("[data-test='open-advanced']").trigger("click");
  await flushPromises();
  const formatControl = wrapper
    .findAllComponents({ name: "SegmentedControl" })
    .find((row) => row.props("label") === "File format");
  return {
    strength: wrapper.text().includes("How much to change it"),
    mask: wrapper.find("[data-test='source-edit-mask']").exists(),
    negative: wrapper.find("[data-test='section-negative']").exists(),
    formats: (formatControl?.props("options") as { value: string }[] | undefined)?.map(
      (option) => option.value,
    ),
  };
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

/**
 * A checkpoint's contract is the checkpoint's, not the machine's. Reading the
 * advertised recipe only from the TARGET host's inventory meant aiming Create
 * at a machine that would download the model first silently replaced the mesh
 * settings with raster ones — Denoise, Mask, Shape, and a Resolution control
 * bound to the canvasless recipe's own 0 × 0, which rendered as `NaN×NaN px`
 * under a "Width and height must be whole numbers" error nobody could clear.
 */
describe("InspectorPanel — a target host that does not have the model", () => {
  it("keeps the canvas hidden for the mesh model it still names", async () => {
    const { wrapper } = mountForTargetWithoutModel(meshModel());
    await flushPromises();
    expect(wrapper.findComponent(ShapePicker).exists()).toBe(false);
    expect(wrapper.findComponent(ResolutionSelector).exists()).toBe(false);
  });

  it("never validates the zero canvas it was told to render", async () => {
    const { wrapper } = mountForTargetWithoutModel(meshModel());
    await flushPromises();
    expect(wrapper.text()).not.toContain("Width and height must be whole numbers");
    expect(wrapper.text()).not.toContain("NaN");
  });

  it("keeps the Mesh group the model's own recipe advertises", async () => {
    const { wrapper } = mountForTargetWithoutModel(meshModel());
    await flushPromises();
    expect(wrapper.find("[data-test='mesh-controls']").exists()).toBe(true);
    const octree = wrapper
      .findAllComponents({ name: "SegmentedControl" })
      .find((row) => row.attributes("data-test") === "mesh-octree")!;
    // Rough | Normal | Fine over the recipe's own floor, default and ceiling.
    expect(octree.props("options")).toEqual([
      { value: 128, label: "Rough" },
      { value: 256, label: "Normal" },
      { value: 384, label: "Fine" },
    ]);
  });

  it("leaves a raster model's canvas alone", async () => {
    const { wrapper } = mountForTargetWithoutModel(rasterModel());
    await flushPromises();
    expect(wrapper.findComponent(ShapePicker).exists()).toBe(true);
    expect(wrapper.findComponent(ResolutionSelector).exists()).toBe(true);
    expect(wrapper.find("[data-test='mesh-controls']").exists()).toBe(false);
  });

  /**
   * The canvas was only the first control that had to follow the fallback
   * recipe. Strength, the repaint mask, the negative prompt and the stored
   * container are asked by `SourceImageWell` and `AdvancedSettings`, which
   * were still handed the TARGET host's row — so switching machines left a
   * Denoise slider, an Edit-mask control, a Negative-prompt field and a
   * png/jpeg/webp picker on a print that has none of them.
   */
  it("offers no strength, mask, negative prompt or raster container", async () => {
    const { wrapper } = mountForTargetWithoutModel(meshModel());
    await flushPromises();
    expect(await panelContract(wrapper)).toEqual({
      strength: false,
      mask: false,
      negative: false,
      formats: ["glb"],
    });
  });

  /** The whole point: the panel reads the same wherever Create is aimed. */
  it("renders the identical contract to the machine that has the model", async () => {
    const onHost = mountForTargetWithModel(meshModel());
    await flushPromises();
    const baseline = await panelContract(onHost.wrapper);
    onHost.wrapper.unmount();
    document.body.innerHTML = "";
    setActivePinia(createPinia());

    const offHost = mountForTargetWithoutModel(meshModel());
    await flushPromises();
    expect(await panelContract(offHost.wrapper)).toEqual(baseline);
    expect(baseline).toEqual({
      strength: false,
      mask: false,
      negative: false,
      formats: ["glb"],
    });
  });

  it("keeps a raster model's strength, mask, negative prompt and formats", async () => {
    const { wrapper } = mountForTargetWithoutModel(rasterModel());
    await flushPromises();
    expect(await panelContract(wrapper)).toEqual({
      strength: true,
      mask: true,
      negative: true,
      formats: ["png", "jpeg", "webp"],
    });
  });

  /**
   * The family rule is only the answer of LAST resort. Where a machine does
   * advertise the checkpoint, the panel must read THAT recipe — including
   * every refusal a family rule would never guess. This checkpoint's own
   * profile pins one container and refuses strength, the mask and the
   * negative prompt, none of which follows from `sdxl`.
   */
  it("reads the advertised recipe's own refusals, not the family defaults", async () => {
    const recipe = sdxlRecipe();
    recipe.capabilities.supports_strength = false;
    recipe.capabilities.mask = { mode: "hidden", required: false };
    recipe.capabilities.negative_prompt = { mode: "hidden", required: false };
    recipe.capabilities.output = {
      default_format: "jpeg",
      formats: ["jpeg"],
      audio_requires_mp4: false,
    };
    const model = modelWith("strict-sdxl:fp16", "sdxl", recipe);
    const { wrapper } = mountForTargetWithoutModel(model);
    await flushPromises();
    expect(await panelContract(wrapper)).toEqual({
      strength: false,
      mask: false,
      negative: false,
      formats: ["jpeg"],
    });
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
      { value: 128, label: "Rough" },
      { value: 256, label: "Normal" },
      { value: 384, label: "Fine" },
    ]);
    // Untouched means "use the profile default", and the default is lit.
    expect(octree.props("modelValue")).toBe(256);
    // The rung itself stays on screen as the mono truth beside the words.
    expect(wrapper.get("[data-test='mesh-octree-truth']").text()).toContain("octree 256");
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
      .find((row) => row.props("label") === "How tight to the photo")!;
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
      .find((row) => row.props("label") === "How tight to the photo")!;
    expect(threshold.props("disabled")).toBe(true);
    expect(wrapper.get("[data-test='mesh-threshold-note']").text()).toBe(
      "This build pins the iso surface.",
    );
  });

  it("bounds Simplify to by the recipe and treats blank as every detail", async () => {
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

  // A budget outside the advertised bounds is a 422 at admission; the
  // inspector names the bounds inline (as it does for Steps) instead of
  // letting Generate learn it from the host.
  it("names the advertised bounds beside a Simplify to value outside them", async () => {
    const { form, wrapper } = mountFor(meshModel());
    await flushPromises();
    const faces = wrapper.get("[data-test='mesh-target-faces']");
    expect(wrapper.find("[data-test='mesh-target-faces-error']").exists()).toBe(false);

    await faces.setValue("10");
    // The typed value stands — it is not snapped behind the user's back.
    expect(form.mesh.targetFaces).toBe(10);
    expect(wrapper.get("[data-test='mesh-target-faces-error']").text()).toBe(
      "Use a whole number of faces from 100 to 2000000.",
    );
    expect(faces.attributes("aria-invalid")).toBe("true");

    await faces.setValue("50000");
    expect(wrapper.find("[data-test='mesh-target-faces-error']").exists()).toBe(false);
    expect(faces.attributes("aria-invalid")).toBeUndefined();
  });
});
