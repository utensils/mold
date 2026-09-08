import { describe, expect, it } from "vitest";
import { supportsMeshWorkflow } from "./meshWorkflowRouting";
import type { WorkflowModel } from "./meshWorkflowAuthoring";

const mesh: WorkflowModel = {
  name: "hunyuan3d",
  family: "hunyuan3d",
  downloaded: true,
  default_steps: 5,
  default_guidance: 5,
  default_width: 0,
  default_height: 0,
  generation_profile: {
    default_recipe_id: "shape",
    recipes: [
      {
        id: "shape",
        capabilities: {
          mesh: {
            workflow_modes: ["text_to_mesh", "mesh_roundtrip", "mesh_texture"],
            delight: { mode: "adjustable" },
          },
        },
      },
    ],
  },
};
const image: WorkflowModel = {
  ...mesh,
  name: "z-image",
  family: "z-image",
  modality: "image",
  generation_profile: null,
};
const request = {
  mode: "text_to_mesh" as const,
  meshModel: mesh.name,
  imageModel: image.name,
  texture: true,
  delight: true,
};

describe("complete mesh workflow placement", () => {
  it("requires the image and mesh stages on the same machine", () => {
    expect(supportsMeshWorkflow([mesh, image], request)).toBe(true);
    expect(supportsMeshWorkflow([mesh], request)).toBe(false);
    expect(supportsMeshWorkflow([image], request)).toBe(false);
  });
  it("refuses unavailable runtimes and missing optional stage capabilities", () => {
    expect(
      supportsMeshWorkflow(
        [{ ...mesh, runtime_available: false }, image],
        request,
      ),
    ).toBe(false);
    const shapeOnly = {
      ...mesh,
      generation_profile: {
        default_recipe_id: "shape",
        recipes: [
          {
            id: "shape",
            capabilities: { mesh: { workflow_modes: ["text_to_mesh"] } },
          },
        ],
      },
    };
    expect(supportsMeshWorkflow([shapeOnly, image], request)).toBe(false);
    expect(
      supportsMeshWorkflow([shapeOnly, image], {
        ...request,
        texture: false,
        delight: false,
      }),
    ).toBe(true);
    expect(
      supportsMeshWorkflow([shapeOnly, image], { ...request, texture: false }),
    ).toBe(false);
  });
  it("does not require an image generator when rebuilding a supplied mesh", () => {
    expect(
      supportsMeshWorkflow([mesh], { ...request, mode: "mesh_roundtrip" }),
    ).toBe(true);
  });
});
