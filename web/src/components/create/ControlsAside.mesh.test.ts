import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import ControlsAside from "./ControlsAside.vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import { __testing__ as routingTesting } from "../../composables/useHostRouting";
import type { GenerateFormState, ModelInfoExtended } from "../../types";
import {
  hunyuan3dRecipe,
  sdxlRecipe,
} from "@studio/lib/generationProfile.testFixtures";

/**
 * The rail's 3-D contract: a canvasless recipe renders no Shape/Resolution at
 * all, and the geometry controls come from the recipe's own `mesh` block —
 * never a client constant.
 */

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../machines/hostClient", () => ({
  hostStatus: () => Promise.reject(new Error("offline in tests")),
  hostModels: () => Promise.reject(new Error("offline in tests")),
  hostCapabilities: () => Promise.reject(new Error("offline in tests")),
  hostQueue: () => Promise.resolve({ entries: [], plan: null }),
  hostDevices: () => Promise.reject(new Error("offline in tests")),
}));

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

function mountMesh(overrides: Partial<GenerateFormState> = {}) {
  return mount(ControlsAside, {
    props: {
      modelValue: baseForm({
        model: "hunyuan3d-mini-turbo:fp16",
        modelFamily: "hunyuan3d",
        width: 0,
        height: 0,
        steps: 5,
        guidance: 5,
        ...overrides,
      }),
      family: "hunyuan3d",
      model: meshModel(),
      advCount: 0,
    },
  });
}

/** The octree ladder's rendered segments, in advertised order. */
function octreeSegments(wrapper: ReturnType<typeof mountMesh>) {
  return wrapper.get("[data-test='mesh-octree']").findAll("[role='radio']");
}

describe("ControlsAside 3-D mesh", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    localStorage.clear();
    routingTesting.reset();
  });
  afterEach(() => __testing__.resetForTest());

  it("hides Shape and Resolution on a canvasless recipe", () => {
    const wrapper = mountMesh();
    expect(wrapper.findComponent(ShapePicker).exists()).toBe(false);
    expect(wrapper.findComponent(ResolutionSelector).exists()).toBe(false);
    expect(wrapper.text()).not.toContain("Shape");
  });

  it("keeps Shape and Resolution on a raster recipe", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ model: "sdxl:fp16", modelFamily: "sdxl" }),
        family: "sdxl",
        model: rasterModel(),
        advCount: 0,
      },
    });
    expect(wrapper.findComponent(ShapePicker).exists()).toBe(true);
    expect(wrapper.findComponent(ResolutionSelector).exists()).toBe(true);
    expect(wrapper.find("[data-test='mesh-controls']").exists()).toBe(false);
  });

  it("offers the advertised octree ladder with the profile default lit", () => {
    const segments = octreeSegments(mountMesh());
    expect(segments.map((segment) => segment.text())).toEqual([
      "128",
      "192",
      "256",
      "320",
      "384",
    ]);
    expect(
      segments.map((segment) => segment.attributes("aria-checked")),
    ).toEqual(["false", "false", "true", "false", "false"]);
  });

  it("records an octree pick on the form", async () => {
    const wrapper = mountMesh();
    await octreeSegments(wrapper)[4]!.trigger("click");
    const emitted = wrapper.emitted("update:modelValue");
    expect(emitted).toBeTruthy();
    const next = emitted!.at(-1)![0] as GenerateFormState;
    expect(next.mesh?.octreeResolution).toBe(384);
  });

  it("binds the iso-threshold slider to the recipe's float control", () => {
    const wrapper = mountMesh();
    const slider = wrapper
      .findAllComponents(SliderRow)
      .find((c) => c.props("label") === "Iso threshold");
    expect(slider).toBeTruthy();
    expect(slider!.props("min")).toBe(0);
    expect(slider!.props("max")).toBe(1);
    expect(slider!.props("step")).toBe(0.01);
    expect(slider!.props("modelValue")).toBe(0.6);
    expect(slider!.props("disabled")).toBe(false);
  });

  it("keeps target faces blank until the user names one", async () => {
    const wrapper = mountMesh();
    const input = wrapper.get("[data-test='mesh-target-faces']");
    expect((input.element as HTMLInputElement).value).toBe("");
    await input.setValue("20000");
    const next = wrapper
      .emitted("update:modelValue")!
      .at(-1)![0] as GenerateFormState;
    expect(next.mesh?.targetFaces).toBe(20_000);
  });

  it("clears target faces back to the raw surface when emptied", async () => {
    const wrapper = mountMesh({
      mesh: { octreeResolution: null, threshold: null, targetFaces: 20_000 },
    });
    const input = wrapper.get("[data-test='mesh-target-faces']");
    expect((input.element as HTMLInputElement).value).toBe("20000");
    await input.setValue("");
    const next = wrapper
      .emitted("update:modelValue")!
      .at(-1)![0] as GenerateFormState;
    expect(next.mesh?.targetFaces).toBeNull();
  });

  // The bounds are the recipe's own; a value outside them is a 422 at
  // admission, so the rail says so inline, the way the resolution warning
  // does, instead of letting Generate fail with no advisory.
  it("warns inline when target faces falls outside the advertised bounds", async () => {
    const wrapper = mountMesh({
      mesh: { octreeResolution: null, threshold: null, targetFaces: 10 },
    });
    const warning = wrapper.get("[data-test='mesh-target-faces-warning']");
    expect(warning.text()).toContain("100");
    expect(warning.text()).toContain("2,000,000");
    expect(warning.text()).toContain("10");
    expect(warning.classes()).toContain("controls__hint--warning");

    await wrapper.setProps({
      modelValue: baseForm({
        model: "hunyuan3d-mini-turbo:fp16",
        modelFamily: "hunyuan3d",
        width: 0,
        height: 0,
        mesh: {
          octreeResolution: null,
          threshold: null,
          targetFaces: 3_000_000,
        },
      }),
    });
    expect(
      wrapper.get("[data-test='mesh-target-faces-warning']").text(),
    ).toContain("3,000,000");
  });

  it("shows no target-faces warning inside the bounds or when blank", () => {
    expect(
      mountMesh({
        mesh: { octreeResolution: null, threshold: null, targetFaces: 20_000 },
      })
        .find("[data-test='mesh-target-faces-warning']")
        .exists(),
    ).toBe(false);
    expect(
      mountMesh().find("[data-test='mesh-target-faces-warning']").exists(),
    ).toBe(false);
  });

  it("renders the server's own note for a fixed iso threshold", () => {
    const recipe = hunyuan3dRecipe();
    recipe.capabilities.mesh!.threshold = {
      ...recipe.capabilities.mesh!.threshold,
      mode: "fixed",
      note: "This build pins the iso surface at 0.60.",
    };
    const model = {
      ...meshModel(),
      generation_profile: {
        schema_version: 1,
        profile_id: "hunyuan3d",
        profile_hash: "h3d",
        default_recipe_id: "default",
        recipes: [recipe],
      },
    } as ModelInfoExtended;
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({
          model: "hunyuan3d-mini-turbo:fp16",
          modelFamily: "hunyuan3d",
          width: 0,
          height: 0,
        }),
        family: "hunyuan3d",
        model,
        advCount: 0,
      },
    });
    expect(wrapper.get("[data-test='mesh-threshold-note']").text()).toContain(
      "pins the iso surface",
    );
    const slider = wrapper
      .findAllComponents(SliderRow)
      .find((c) => c.props("label") === "Iso threshold")!;
    expect(slider.props("disabled")).toBe(true);
  });
});
