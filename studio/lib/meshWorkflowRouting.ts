import type { ApiTarget } from "../api/client";
import { isMeshFamily } from "./legacyRecipeRules";
import {
  isTextImageWorkflowModel,
  meshWorkflowModes,
  type WorkflowModel,
} from "./meshWorkflowAuthoring";

export interface MeshWorkflowRequirements {
  mode: "text_to_mesh" | "mesh_roundtrip" | "mesh_texture";
  meshModel: string;
  imageModel?: string;
  texture: boolean;
  delight: boolean;
}

export interface MeshWorkflowRoute {
  target: ApiTarget;
  label: string;
}

/** All stages stay on one machine; a fleet-wide union is not an execution plan. */
export function supportsMeshWorkflow(
  models: readonly WorkflowModel[],
  request: MeshWorkflowRequirements,
): boolean {
  const mesh = models.find((model) => model.name === request.meshModel);
  if (
    !mesh?.downloaded ||
    mesh.runtime_available === false ||
    !isMeshFamily(mesh.family)
  )
    return false;
  const modes = meshWorkflowModes(mesh);
  if (
    !modes.includes(request.mode) ||
    (request.texture && !modes.includes("mesh_texture"))
  )
    return false;
  if (request.delight) {
    const profile = mesh.generation_profile;
    const recipe = profile?.recipes.find(
      (value) => value.id === profile.default_recipe_id,
    );
    if (recipe?.capabilities.mesh?.delight?.mode !== "adjustable") return false;
  }
  return (
    request.mode !== "text_to_mesh" ||
    models.some(
      (model) =>
        model.name === request.imageModel && isTextImageWorkflowModel(model),
    )
  );
}
