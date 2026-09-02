/**
 * The desktop form on a Hunyuan3D recipe: a canvasless mesh request is
 * pinned to glb, carries no canvas-only fields, and round-trips the mesh
 * controls through the request and the print's metadata.
 */
import { describe, expect, it } from "vitest";
import { hunyuan3dRecipe, sdxlRecipe } from "@studio/lib/generationProfile.testFixtures";
import type { GenerationProfileSet } from "@studio/lib/generationProfile";
import type { ModelEntry, OutputMetadata } from "./api/types";
import { pruneRequestForFamily, recipeCapabilitiesSnapshot } from "./capabilities";
import {
  applyMetadataToForm,
  applyModelDefaults,
  applyRequestToForm,
  buildRequest,
  newGenerateForm,
} from "./generateForm";

function profile(id: string, recipe: ReturnType<typeof hunyuan3dRecipe>): GenerationProfileSet {
  return {
    schema_version: 1,
    profile_id: id,
    profile_hash: `${id}-hash`,
    default_recipe_id: recipe.id,
    recipes: [recipe],
  };
}

function hunyuanModel(): ModelEntry {
  return {
    name: "hunyuan3d-mini-turbo:fp16",
    family: "hunyuan3d",
    size_gb: 2,
    is_loaded: false,
    hf_repo: "tencent/Hunyuan3D-2mini",
    default_steps: 5,
    default_guidance: 5,
    default_width: 1024,
    default_height: 1024,
    description: "",
    downloaded: true,
    source_image: "required",
    generation_profile: profile("hunyuan3d.mini", hunyuan3dRecipe()),
  };
}

function sdxlModel(): ModelEntry {
  return {
    name: "sdxl-base:fp16",
    family: "sdxl",
    size_gb: 6,
    is_loaded: false,
    hf_repo: "sdxl",
    default_steps: 30,
    default_guidance: 7,
    default_width: 1024,
    default_height: 1024,
    description: "",
    downloaded: true,
    generation_profile: profile("sdxl.base", sdxlRecipe()),
  };
}

function meshForm() {
  const form = newGenerateForm();
  form.model = hunyuanModel().name;
  form.family = "hunyuan3d";
  applyModelDefaults(form, hunyuanModel());
  form.sourceImage = "c291cmNl";
  form.sourceImageName = "armchair.png";
  return form;
}

describe("selecting a Hunyuan3D recipe", () => {
  it("pins glb, takes the recipe's zero canvas, and snapshots the mesh controls", () => {
    const form = meshForm();
    expect(form.outputFormat).toBe("glb");
    expect(form.width).toBe(0);
    expect(form.height).toBe(0);
    expect(form.recipeCapabilities?.canvasless).toBe(true);
    expect(form.recipeCapabilities?.mesh?.octree_default).toBe(256);
    expect(form.recipeCapabilities?.promptMode).toBe("ignored");
    expect(form.recipeCapabilities?.supportsStrength).toBe(false);
  });

  it("clears the mesh controls and restores a raster format when leaving the family", () => {
    const form = meshForm();
    form.mesh.octreeResolution = 192;
    form.model = sdxlModel().name;
    form.family = "sdxl";
    applyModelDefaults(form, sdxlModel());
    expect(form.outputFormat).toBe("png");
    expect(form.mesh).toEqual({ octreeResolution: null, threshold: null, targetFaces: null });
    expect(form.recipeCapabilities?.mesh).toBeNull();
    expect(form.recipeCapabilities?.canvasless).toBe(false);
  });
});

describe("buildRequest on a Hunyuan3D recipe", () => {
  it("sends glb with a zero canvas and no strength, mask, or source_fit", () => {
    const form = meshForm();
    form.maskImage = "bWFzaw==";
    const req = buildRequest(form);
    expect(req.output_format).toBe("glb");
    expect(req.width).toBe(0);
    expect(req.height).toBe(0);
    expect(req.source_image).toBe("c291cmNl");
    expect(req).not.toHaveProperty("strength");
    expect(req).not.toHaveProperty("mask_image");
    expect(req).not.toHaveProperty("source_fit");
    expect(req).not.toHaveProperty("mesh");
  });

  it("submits without a prompt because the recipe ignores it", () => {
    const form = meshForm();
    form.prompt = "";
    expect(buildRequest(form).prompt).toBe("");
  });

  it("carries only the mesh controls that differ from the advertised defaults", () => {
    const form = meshForm();
    form.mesh = { octreeResolution: 192, threshold: 0.6, targetFaces: 50_000 };
    expect(buildRequest(form).mesh).toEqual({ octree_resolution: 192, target_faces: 50_000 });
  });

  it("pins a raster format an older snapshot left behind", () => {
    const form = meshForm();
    form.outputFormat = "png";
    expect(buildRequest(form).output_format).toBe("glb");
  });
});

describe("pruneRequestForFamily with a recipe", () => {
  it("pins png to glb for a mesh recipe and drops canvas-only fields", () => {
    const pruned = pruneRequestForFamily(
      {
        prompt: "",
        model: "hunyuan3d-mini-turbo:fp16",
        width: 0,
        height: 0,
        steps: 5,
        output_format: "png",
        source_image: "c291cmNl",
        strength: 0.75,
        mesh: { octree_resolution: 192 },
      },
      "hunyuan3d",
      "hunyuan3d-mini-turbo:fp16",
      "required",
      recipeCapabilitiesSnapshot(hunyuan3dRecipe()),
    );
    expect(pruned.output_format).toBe("glb");
    expect(pruned.mesh).toEqual({ octree_resolution: 192 });
    expect(pruned).not.toHaveProperty("strength");
  });

  it("drops a lingering mesh block and glb on a raster recipe", () => {
    const pruned = pruneRequestForFamily(
      {
        prompt: "a cat",
        model: "sdxl-base:fp16",
        width: 1024,
        height: 1024,
        steps: 30,
        output_format: "glb",
        mesh: { octree_resolution: 192 },
      },
      "sdxl",
      "sdxl-base:fp16",
      null,
      recipeCapabilitiesSnapshot(sdxlRecipe()),
    );
    expect(pruned.output_format).toBe("png");
    expect(pruned).not.toHaveProperty("mesh");
  });

  it("keeps the legacy family rule when no recipe is known", () => {
    const pruned = pruneRequestForFamily(
      {
        prompt: "",
        model: "hunyuan3d-mini-turbo:fp16",
        width: 0,
        height: 0,
        steps: 5,
        output_format: "png",
      },
      "hunyuan3d",
      "hunyuan3d-mini-turbo:fp16",
    );
    expect(pruned.output_format).toBe("glb");
  });
});

describe("restoring a mesh print", () => {
  const metadata: OutputMetadata = {
    prompt: "",
    model: "hunyuan3d-mini-turbo:fp16",
    seed: 7,
    steps: 5,
    guidance: 5,
    width: 512,
    height: 512,
    output_format: "png",
    mesh: { octree_resolution: 320, threshold: 0.55, target_faces: 40_000 },
  };

  it("applyMetadataToForm restores the mesh controls and pins glb", () => {
    const form = newGenerateForm();
    applyMetadataToForm(form, metadata, [hunyuanModel()]);
    expect(form.mesh).toEqual({ octreeResolution: 320, threshold: 0.55, targetFaces: 40_000 });
    expect(form.outputFormat).toBe("glb");
    expect(form.width).toBe(0);
    expect(form.height).toBe(0);
  });

  it("applyRequestToForm restores the mesh controls and pins glb", () => {
    const form = newGenerateForm();
    applyRequestToForm(
      form,
      {
        prompt: "",
        model: "hunyuan3d-mini-turbo:fp16",
        width: 0,
        height: 0,
        steps: 5,
        output_format: "png",
        mesh: { octree_resolution: 192 },
      },
      [hunyuanModel()],
    );
    expect(form.mesh).toEqual({ octreeResolution: 192, threshold: null, targetFaces: null });
    expect(form.outputFormat).toBe("glb");
  });

  it("leaves the mesh controls empty for a raster print", () => {
    const form = newGenerateForm();
    applyMetadataToForm(
      form,
      { ...metadata, model: "sdxl-base:fp16", output_format: "png", mesh: null },
      [sdxlModel()],
    );
    expect(form.mesh).toEqual({ octreeResolution: null, threshold: null, targetFaces: null });
    expect(form.outputFormat).toBe("png");
  });
});
