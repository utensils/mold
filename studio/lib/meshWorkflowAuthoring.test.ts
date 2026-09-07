import { describe, expect, it } from "vitest";

import {
  buildMeshTextureWorkflow,
  buildTextToMeshWorkflow,
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
});
