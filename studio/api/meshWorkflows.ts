import { apiFetchTo, apiJsonTo, type ApiTarget } from "./client";

export type MeshWorkflowJobState =
  "queued" | "running" | "paused" | "completed" | "failed" | "cancelled";

export type MeshWorkflowStageKind =
  "image" | "matting" | "delight" | "shape" | "paint" | "finalize";

export type MeshWorkflowStageState =
  "pending" | "running" | "completed" | "failed";

export interface MeshWorkflowArtifact {
  role: string;
  relative_path: string;
  media_type: string;
  sha256: string;
  byte_length: number;
}

export interface MeshWorkflowStageRecord {
  index: number;
  kind: MeshWorkflowStageKind;
  state: MeshWorkflowStageState;
  execution_batch_id?: string;
  artifacts?: MeshWorkflowArtifact[];
  error?: string;
}

export type CreateMeshWorkflowRequest<TRequest = unknown> =
  | {
      mode: "text_to_mesh";
      image_request: TRequest;
      mesh_request: TRequest;
    }
  | { mode: "mesh_roundtrip"; roundtrip_request: TRequest }
  | { mode: "mesh_texture"; texture_request: TRequest };

export interface CreateMeshWorkflowResponse {
  job_id: string;
  request_warnings?: string[];
}

export interface MeshWorkflowJobSummary {
  contract_version: number;
  id: string;
  state: MeshWorkflowJobState;
  mode: "text_to_mesh" | "mesh_roundtrip" | "mesh_texture";
  stage_count: number;
  current_stage: number;
  current_stage_kind?: MeshWorkflowStageKind;
  output_filename?: string;
  error?: string;
  created_at_ms: number;
  updated_at_ms: number;
}

export interface MeshWorkflowJobDetail<
  TRequest = unknown,
> extends MeshWorkflowJobSummary {
  request: CreateMeshWorkflowRequest<TRequest>;
  stages: MeshWorkflowStageRecord[];
}

export interface MeshWorkflowJobListing {
  jobs: MeshWorkflowJobSummary[];
}

export type MeshWorkflowEvent<TRequest = unknown> = {
  event: "snapshot";
  job: MeshWorkflowJobDetail<TRequest>;
};

export function createMeshWorkflow<TRequest>(
  target: ApiTarget,
  request: CreateMeshWorkflowRequest<TRequest>,
): Promise<CreateMeshWorkflowResponse> {
  return apiJsonTo(target, "/api/mesh-workflows", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
  });
}

export function listMeshWorkflows(
  target: ApiTarget,
): Promise<MeshWorkflowJobListing> {
  return apiJsonTo(target, "/api/mesh-workflows");
}

export function getMeshWorkflow<TRequest = unknown>(
  target: ApiTarget,
  id: string,
): Promise<MeshWorkflowJobDetail<TRequest>> {
  return apiJsonTo(target, `/api/mesh-workflows/${encodeURIComponent(id)}`);
}

export async function resumeMeshWorkflow(
  target: ApiTarget,
  id: string,
): Promise<void> {
  await apiFetchTo(
    target,
    `/api/mesh-workflows/${encodeURIComponent(id)}/resume`,
    {
      method: "POST",
    },
  );
}

export async function cancelMeshWorkflow(
  target: ApiTarget,
  id: string,
): Promise<void> {
  await apiFetchTo(
    target,
    `/api/mesh-workflows/${encodeURIComponent(id)}/cancel`,
    {
      method: "POST",
    },
  );
}

export async function deleteMeshWorkflow(
  target: ApiTarget,
  id: string,
): Promise<void> {
  await apiFetchTo(target, `/api/mesh-workflows/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export function meshWorkflowEventsUrl(target: ApiTarget, id: string): string {
  return `${target.baseUrl}/api/mesh-workflows/${encodeURIComponent(id)}/events`;
}
