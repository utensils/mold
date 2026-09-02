import { beforeEach, describe, expect, it } from "vitest";
import { IDBFactory } from "fake-indexeddb";
import {
  applyMetadataToForm,
  useGenerateForm,
  __testing__,
} from "./useGenerateForm";
import type { ModelInfoExtended, OutputMetadata } from "../types";
import {
  hunyuan3dRecipe,
  sdxlRecipe,
} from "@studio/lib/generationProfile.testFixtures";

/**
 * The 3-D half of the Create form contract: a mesh recipe pins the output
 * format to GLB, renders on no canvas at all, and carries its geometry
 * controls in `mesh` — while every raster-only field (fit policy, strength,
 * mask) stays off the wire.
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

describe("useGenerateForm mesh recipes", () => {
  beforeEach(() => {
    Object.defineProperty(globalThis, "indexedDB", {
      value: new IDBFactory(),
      writable: true,
      configurable: true,
    });
    localStorage.clear();
    __testing__.resetForTest();
  });

  it("pins the output format to glb and the canvas to the recipe defaults", () => {
    const form = useGenerateForm();
    form.state.value.outputFormat = "png";
    form.applyModelDefaults(meshModel());
    expect(form.state.value.outputFormat).toBe("glb");
    expect(form.state.value.width).toBe(0);
    expect(form.state.value.height).toBe(0);
  });

  it("restores a raster format and clears the mesh form on the way back", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(meshModel());
    form.state.value.mesh = {
      octreeResolution: 384,
      threshold: 0.42,
      targetFaces: 20_000,
    };
    form.applyModelDefaults(rasterModel());
    expect(form.state.value.outputFormat).toBe("png");
    expect(form.state.value.width).toBe(1024);
    expect(form.state.value.mesh).toEqual({
      octreeResolution: null,
      threshold: null,
      targetFaces: null,
    });
  });

  it("builds a mesh request with no raster-only fields", () => {
    const form = useGenerateForm();
    const model = meshModel();
    form.applyModelDefaults(model);
    form.state.value.prompt = "";
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "chair.png", base64: "AAA" },
    ];
    form.state.value.maskImage = {
      kind: "upload",
      filename: "mask.png",
      base64: "MMM",
    };
    const request = form.toRequest(model);
    expect(request.output_format).toBe("glb");
    expect(request.width).toBe(0);
    expect(request.height).toBe(0);
    expect(request.source_fit).toBeUndefined();
    expect(request.strength).toBeUndefined();
    expect(request.mask_image).toBeUndefined();
    // Nothing differs from the advertised defaults, so no mesh block travels.
    expect(request.mesh).toBeUndefined();
  });

  it("sends only the mesh controls that differ from the recipe defaults", () => {
    const form = useGenerateForm();
    const model = meshModel();
    form.applyModelDefaults(model);
    form.state.value.mesh = {
      // 256 IS the advertised default, so it stays off the wire.
      octreeResolution: 256,
      threshold: 0.42,
      targetFaces: 20_000,
    };
    expect(form.toRequest(model).mesh).toEqual({
      threshold: 0.42,
      target_faces: 20_000,
    });
  });

  it("never sends a mesh block on a raster recipe", () => {
    const form = useGenerateForm();
    const model = rasterModel();
    form.applyModelDefaults(model);
    form.state.value.mesh = {
      octreeResolution: 384,
      threshold: 0.42,
      targetFaces: null,
    };
    expect(form.toRequest(model).mesh).toBeUndefined();
  });

  it("coerces a stale raster format to glb at request time", () => {
    const form = useGenerateForm();
    const model = meshModel();
    form.applyModelDefaults(model);
    // A draft restored from a pre-mesh snapshot can still hold `png`.
    form.state.value.outputFormat = "png";
    expect(form.toRequest(model).output_format).toBe("glb");
  });

  it("restores the mesh controls and the glb format from a mesh print", () => {
    const form = useGenerateForm();
    const next = applyMetadataToForm(
      form.state.value,
      {
        prompt: "",
        model: "hunyuan3d-mini-turbo:fp16",
        seed: 7,
        steps: 5,
        guidance: 5,
        width: 1024,
        height: 1024,
        output_format: "png",
        mesh: {
          octree_resolution: 384,
          threshold: 0.42,
          target_faces: 20_000,
          texture: false,
        },
      } as OutputMetadata,
      { models: [meshModel()] },
    );
    expect(next.outputFormat).toBe("glb");
    expect(next.width).toBe(0);
    expect(next.height).toBe(0);
    expect(next.mesh).toEqual({
      octreeResolution: 384,
      threshold: 0.42,
      targetFaces: 20_000,
    });
  });

  it("drops a restored mesh form when the reused print is raster", () => {
    const form = useGenerateForm();
    form.state.value.mesh = {
      octreeResolution: 384,
      threshold: 0.42,
      targetFaces: null,
    };
    const next = applyMetadataToForm(
      form.state.value,
      {
        prompt: "a lighthouse",
        model: "sdxl:fp16",
        seed: 7,
        steps: 25,
        guidance: 7,
        width: 1024,
        height: 1024,
      } as OutputMetadata,
      { models: [rasterModel()] },
    );
    expect(next.mesh).toEqual({
      octreeResolution: null,
      threshold: null,
      targetFaces: null,
    });
    expect(next.outputFormat).toBe("png");
  });
});
