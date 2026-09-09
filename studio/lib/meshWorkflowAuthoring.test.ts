import { describe, expect, it } from "vitest";

import {
  buildMeshRoundtripWorkflow,
  buildMeshTextureWorkflow,
  buildTextToMeshWorkflow,
  isTextImageWorkflowModel,
  meshWorkflowModes,
  type WorkflowModel,
} from "./meshWorkflowAuthoring";

function model(
  name: string,
  family: string,
  modes: string[] = [],
): WorkflowModel {
  return {
    name,
    family,
    downloaded: true,
    default_steps: family === "hunyuan3d" ? 30 : 4,
    default_guidance: family === "hunyuan3d" ? 5 : 1,
    default_width: family === "hunyuan3d" ? 0 : 1024,
    default_height: family === "hunyuan3d" ? 0 : 1024,
    generation_profile: {
      default_recipe_id: "default",
      recipes: [
        {
          id: "default",
          capabilities: { mesh: { workflow_modes: modes } },
        },
      ],
    },
  };
}

describe("the picture-style candidates", () => {
  /*
   * `modality` is a CATALOG field: `model_manager` fills it only from a
   * `cv:`/`hf:` sidecar, so every manifest checkpoint answers `undefined` and
   * `modality !== "video"` waved LTX-2, Wan and MiniMax H3 straight into the
   * Picture style list. The one partition authority is `outputKindForModel`,
   * which the New image section strip and the Styles kind filter already read.
   */
  it("refuses a manifest video style that never reported a modality", () => {
    for (const family of ["ltx-2", "ltx-video", "wan", "minimax-h3"]) {
      const clip = model(`${family}-checkpoint`, family);
      expect(clip.modality).toBeUndefined();
      expect(isTextImageWorkflowModel(clip), family).toBe(false);
    }
  });

  it("still offers ordinary still-picture styles", () => {
    for (const family of ["flux", "z-image", "sdxl", "qwen-image"])
      expect(isTextImageWorkflowModel(model(`${family}-checkpoint`, family)), family).toBe(true);
  });

  it("refuses the mesh family, an undownloaded style and an unrunnable one", () => {
    expect(isTextImageWorkflowModel(model("h3", "hunyuan3d", ["text_to_mesh"]))).toBe(false);
    expect(isTextImageWorkflowModel({ ...model("flux", "flux"), downloaded: false })).toBe(false);
    expect(
      isTextImageWorkflowModel({ ...model("flux", "flux"), runtime_available: false }),
    ).toBe(false);
  });

  it("refuses a canvasless recipe and one that ignores the prompt", () => {
    const canvasless = model("odd", "flux");
    canvasless.generation_profile!.recipes[0]!.capabilities.canvasless = true;
    expect(isTextImageWorkflowModel(canvasless)).toBe(false);
    const ignored = model("mute", "flux");
    ignored.generation_profile!.recipes[0]!.capabilities.prompt = { mode: "ignored" };
    expect(isTextImageWorkflowModel(ignored)).toBe(false);
  });
});

describe("mesh workflow authoring", () => {
  it("reads workflow availability from the selected recipe contract", () => {
    expect(
      meshWorkflowModes(model("h3", "hunyuan3d", ["text_to_mesh"])),
    ).toEqual(["text_to_mesh"]);
    expect(meshWorkflowModes(model("old", "hunyuan3d"))).toEqual([]);
  });

  it("keeps the text prompt in the image stage and reserves GLB for the mesh stage", () => {
    const request = buildTextToMeshWorkflow({
      prompt: "a carved wooden fox",
      imageModel: model("flux", "flux"),
      meshModel: model("h3", "hunyuan3d"),
      texture: true,
      textureResolution: 2048,
    });
    expect(request.image_request).toMatchObject({
      prompt: "a carved wooden fox",
      model: "flux",
      width: 1024,
      output_format: "png",
    });
    expect(request.mesh_request).toMatchObject({
      prompt: "",
      model: "h3",
      width: 0,
      output_format: "glb",
      mesh: { texture: true, texture_resolution: 2048 },
    });
    expect(request.mesh_request.source_image).toBeUndefined();
  });

  it("authors distinct mesh and appearance authorities for texture-only work", () => {
    const request = buildMeshTextureWorkflow({
      meshModel: model("h3", "hunyuan3d", ["mesh_texture"]),
      meshBase64: "Z2xi",
      meshName: "chair.glb",
      meshByteLength: 3,
      meshSha256: "ab".repeat(32),
      meshFormat: "glb",
      appearanceBase64: "cG5n",
      upAxis: "z",
      metersPerUnit: 0.001,
      textureResolution: 4096,
    });
    expect(request.texture_request.source_image).toBe("cG5n");
    expect(request.texture_request.references).toEqual([
      {
        kind: "mesh",
        media: { authority: "inline", data: "Z2xi" },
        mime_type: "model/gltf-binary",
        format: "glb",
        byte_length: 3,
        coordinates: { up_axis: "z", meters_per_unit: 0.001 },
        provenance: { name: "chair.glb", sha256: "ab".repeat(32) },
      },
    ]);
  });

  it("can author a payload-free mesh descriptor for streaming upload", () => {
    const request = buildMeshTextureWorkflow({
      meshModel: model("h3", "hunyuan3d", ["mesh_texture"]),
      meshName: "large.glb",
      meshByteLength: 256 * 1024 * 1024,
      meshFormat: "glb",
      appearanceBase64: "cG5n",
      upAxis: "y",
      metersPerUnit: 1,
      textureResolution: 2048,
    });
    expect(request.texture_request.references?.[0]).toMatchObject({
      media: { authority: "descriptor" },
      byte_length: 256 * 1024 * 1024,
      provenance: { name: "large.glb" },
    });
    expect(JSON.stringify(request)).not.toContain("meshBase64");
  });

  it("authors a promptless 2.1 mesh round-trip without appearance conditioning", () => {
    const request = buildMeshRoundtripWorkflow({
      meshModel: model("hunyuan3d-2.1:fp16", "hunyuan3d", ["mesh_roundtrip"]),
      meshBase64: "Z2xi",
      meshName: "scan.glb",
      meshByteLength: 3,
      meshSha256: "cd".repeat(32),
      meshFormat: "glb",
      upAxis: "y",
      metersPerUnit: 1,
    });
    expect(request).toMatchObject({
      mode: "mesh_roundtrip",
      roundtrip_request: {
        prompt: "",
        model: "hunyuan3d-2.1:fp16",
        width: 0,
        height: 0,
        batch_size: 1,
        output_format: "glb",
        mesh: { texture: false },
      },
    });
    expect(request.roundtrip_request.source_image).toBeUndefined();
    expect(request.roundtrip_request.references).toHaveLength(1);
  });
});
