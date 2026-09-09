import type { CreateMeshWorkflowRequest } from "../api/meshWorkflows";
import type { GenerationReference } from "./generationReferences";
import { outputKindForModel } from "./outputKind";

export type MeshWorkflowMode =
  | "image_to_mesh"
  | "multiview_to_mesh"
  | "mesh_roundtrip"
  | "mesh_texture"
  | "text_to_mesh";

export interface WorkflowModel {
  name: string;
  family: string;
  downloaded: boolean;
  display_name?: string | null;
  modality?: string | null;
  runtime_available?: boolean | null;
  default_steps: number;
  default_guidance: number;
  default_width: number;
  default_height: number;
  generation_profile?: {
    default_recipe_id: string;
    recipes: Array<{
      id: string;
      capabilities: {
        canvasless?: boolean;
        prompt?: { mode?: string };
        mesh?: {
          workflow_modes?: string[];
          delight?: { mode?: string } | null;
        } | null;
      };
    }>;
  } | null;
}

export interface WorkflowGenerateRequest {
  prompt: string;
  model: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: null;
  batch_size: 1;
  output_format: "png" | "glb";
  source_image?: string;
  references?: GenerationReference[];
  mesh?: {
    texture?: boolean;
    texture_resolution?: number;
    delight?: true;
  };
}

function defaultRecipe(model: WorkflowModel) {
  const profile = model.generation_profile;
  if (!profile) return null;
  return (
    profile.recipes.find((recipe) => recipe.id === profile.default_recipe_id) ??
    profile.recipes[0] ??
    null
  );
}

export function meshWorkflowModes(model: WorkflowModel): MeshWorkflowMode[] {
  const values = defaultRecipe(model)?.capabilities.mesh?.workflow_modes ?? [];
  return values.filter((value): value is MeshWorkflowMode =>
    [
      "image_to_mesh",
      "multiview_to_mesh",
      "mesh_roundtrip",
      "mesh_texture",
      "text_to_mesh",
    ].includes(value),
  );
}

/**
 * Which styles may drive the text-to-3-D image stage.
 *
 * The kind test is `outputKindForModel` — the ONE partition the New image
 * section strip and the Styles kind filter already read — so this list can
 * never disagree with the rest of the app about what a picture style is. It
 * used to ask `family !== "hunyuan3d" && modality !== "video"`, and `modality`
 * is a CATALOG field that `model_manager` fills only from a `cv:`/`hf:`
 * sidecar: every manifest checkpoint answers `undefined`, so `undefined !==
 * "video"` waved LTX-2, Wan and MiniMax H3 into the Picture style menu.
 *
 * The recipe tests stay on top of the partition and are this stage's own: it
 * hands the stage a prompt and takes a picture back, so a canvasless recipe
 * and one whose prompt is `ignored` are refused even though both are stills.
 */
export function isTextImageWorkflowModel(model: WorkflowModel): boolean {
  const recipe = defaultRecipe(model);
  return (
    model.downloaded &&
    model.runtime_available !== false &&
    outputKindForModel(model) === "still" &&
    recipe?.capabilities.canvasless !== true &&
    recipe?.capabilities.prompt?.mode !== "ignored"
  );
}

function requestFor(
  model: WorkflowModel,
  prompt: string,
  output: "png" | "glb",
) {
  return {
    prompt,
    model: model.name,
    width: output === "glb" ? 0 : model.default_width,
    height: output === "glb" ? 0 : model.default_height,
    steps: model.default_steps,
    guidance: model.default_guidance,
    seed: null,
    batch_size: 1 as const,
    output_format: output,
  };
}

export function buildTextToMeshWorkflow(options: {
  prompt: string;
  imageModel: WorkflowModel;
  meshModel: WorkflowModel;
  texture: boolean;
  textureResolution: number;
  delight?: boolean;
}): Extract<
  CreateMeshWorkflowRequest<WorkflowGenerateRequest>,
  { mode: "text_to_mesh" }
> {
  const mesh = requestFor(
    options.meshModel,
    "",
    "glb",
  ) as WorkflowGenerateRequest;
  if (options.texture) {
    mesh.mesh = {
      texture: true,
      texture_resolution: options.textureResolution,
      ...(options.delight ? { delight: true as const } : {}),
    };
  } else if (options.delight) {
    mesh.mesh = { delight: true };
  }
  return {
    mode: "text_to_mesh",
    image_request: requestFor(options.imageModel, options.prompt.trim(), "png"),
    mesh_request: mesh,
  };
}

export function buildMeshTextureWorkflow(options: {
  meshModel: WorkflowModel;
  meshBase64?: string;
  meshName: string;
  meshByteLength: number;
  meshSha256?: string;
  meshFormat: "glb" | "obj";
  appearanceBase64: string;
  upAxis: "y" | "z";
  metersPerUnit: number;
  textureResolution: number;
  delight?: boolean;
}): Extract<
  CreateMeshWorkflowRequest<WorkflowGenerateRequest>,
  { mode: "mesh_texture" }
> {
  const request = requestFor(
    options.meshModel,
    "",
    "glb",
  ) as WorkflowGenerateRequest;
  request.source_image = options.appearanceBase64;
  request.mesh = {
    texture: true,
    texture_resolution: options.textureResolution,
    ...(options.delight ? { delight: true } : {}),
  };
  request.references = [
    {
      kind: "mesh",
      media: options.meshBase64
        ? { authority: "inline", data: options.meshBase64 }
        : { authority: "descriptor" },
      mime_type:
        options.meshFormat === "glb" ? "model/gltf-binary" : "model/obj",
      format: options.meshFormat,
      byte_length: options.meshByteLength,
      coordinates: {
        up_axis: options.upAxis,
        meters_per_unit: options.metersPerUnit,
      },
      provenance: {
        name: options.meshName,
        ...(options.meshSha256 ? { sha256: options.meshSha256 } : {}),
      },
    },
  ];
  return { mode: "mesh_texture", texture_request: request };
}

export function buildMeshRoundtripWorkflow(options: {
  meshModel: WorkflowModel;
  meshBase64?: string;
  meshName: string;
  meshByteLength: number;
  meshSha256?: string;
  meshFormat: "glb" | "obj";
  upAxis: "y" | "z";
  metersPerUnit: number;
}): Extract<
  CreateMeshWorkflowRequest<WorkflowGenerateRequest>,
  { mode: "mesh_roundtrip" }
> {
  const request = requestFor(
    options.meshModel,
    "",
    "glb",
  ) as WorkflowGenerateRequest;
  request.mesh = { texture: false };
  request.references = [
    {
      kind: "mesh",
      media: options.meshBase64
        ? { authority: "inline", data: options.meshBase64 }
        : { authority: "descriptor" },
      mime_type:
        options.meshFormat === "glb" ? "model/gltf-binary" : "model/obj",
      format: options.meshFormat,
      byte_length: options.meshByteLength,
      coordinates: {
        up_axis: options.upAxis,
        meters_per_unit: options.metersPerUnit,
      },
      provenance: {
        name: options.meshName,
        ...(options.meshSha256 ? { sha256: options.meshSha256 } : {}),
      },
    },
  ];
  return { mode: "mesh_roundtrip", roundtrip_request: request };
}
