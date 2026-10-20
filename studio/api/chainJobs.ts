import { ApiError, apiFetchTo, apiJsonTo, type ApiTarget } from "./client";
import type { ChainJobDetail, ChainRequestWire } from "../lib/api/chainTypes";

export interface ChainLimitsWire {
  model: string;
  supports_sequence: boolean;
  sequence_unsupported_reason?: string | null;
  frames_per_clip_cap: number;
  frames_per_clip_recommended: number;
  max_stages: number;
  supports_audio: boolean;
  transition_modes: string[];
  linked_parts?: {
    supported: boolean;
    max_stages_per_part: number;
    max_parts: number | null;
  };
  authoring?: {
    motion_tail_frames: number;
    carry_mode: "latent" | "rgb" | "independent";
    min_frames: number;
    frame_step: number;
    frame_offset: number;
    source_fit_supported?: boolean;
  };
}
export interface ChainPlan {
  motion_tail_frames: number;
  estimated_total_frames: number;
  estimated_duration_ms: number;
  fps: number;
  warnings: string[];
}
export interface ChainOwner {
  hostId: string;
  label: string;
  instanceId?: string | null | undefined;
  target: ApiTarget;
}
export const CHAIN_ADMISSION_KEY = "mold.video-clips.admissions.v1";
export interface ChainAdmission {
  operationId: string;
  hostId: string;
  instanceId: string;
  baseUrl: string;
  connectionMode?: string;
  createdAt: number;
  cancelRequested?: boolean;
  cancelAcknowledged?: boolean;
}
export function definiteChainRefusal(error: unknown): boolean {
  return (
    error instanceof ApiError &&
    [400, 401, 403, 404, 413, 415, 422, 451].includes(error.status)
  );
}
export function sameChainOwner(
  admission: ChainAdmission,
  owner: ChainOwner,
): boolean {
  return (
    admission.hostId === owner.hostId &&
    admission.instanceId === owner.instanceId &&
    admission.baseUrl === owner.target.baseUrl &&
    admission.connectionMode === owner.target.connectionMode
  );
}
export async function assertChainOwner(owner: ChainOwner): Promise<void> {
  if (!owner.instanceId)
    throw new Error("Reconnect this machine before making video clips.");
  const status = await apiJsonTo<{ instance_id: string }>(
    owner.target,
    "/api/status",
  );
  if (status.instance_id !== owner.instanceId)
    throw new Error(
      "This machine installation changed. Reconnect it before continuing.",
    );
}
/** A Library origin may have no persisted UUID. Capture it from this exact
 * authenticated route on an explicit open, then fence every later action. */
export async function captureChainOwner(
  owner: ChainOwner,
): Promise<ChainOwner> {
  const status = await apiJsonTo<{ instance_id?: string }>(
    owner.target,
    "/api/status",
  );
  if (
    !status.instance_id ||
    (owner.instanceId && status.instance_id !== owner.instanceId)
  )
    throw new Error(
      "This machine installation changed or cannot identify itself. Reconnect before continuing.",
    );
  return {
    ...owner,
    instanceId: status.instance_id,
    target: { ...owner.target },
  };
}
export function fetchChainLimits(
  owner: ChainOwner,
  model: string,
  fps: number,
): Promise<ChainLimitsWire> {
  return apiJsonTo(
    owner.target,
    `/api/capabilities/chain-limits?model=${encodeURIComponent(model)}&fps=${fps}`,
  );
}
/** The server decodes legacy shell newline escapes once at its input boundary.
 * Protect literal backslashes on a wire COPY, matching mold_core::prompt_text. */
function chainBody(request: ChainRequestWire): string {
  const protect = (prompt: string) => prompt.replaceAll("\\", "\\\\");
  return JSON.stringify({
    ...request,
    ...(request.prompt != null ? { prompt: protect(request.prompt) } : {}),
    ...(request.original_prompt != null
      ? { original_prompt: protect(request.original_prompt) }
      : {}),
    ...(request.prompt_transform
      ? {
          prompt_transform: {
            ...request.prompt_transform,
            source_prompt: protect(request.prompt_transform.source_prompt),
            ...(request.prompt_transform.root_prompt != null
              ? { root_prompt: protect(request.prompt_transform.root_prompt) }
              : {}),
          },
        }
      : {}),
    ...(request.stages
      ? {
          stages: request.stages.map((stage) => ({
            ...stage,
            prompt: protect(stage.prompt),
            ...(stage.negative_prompt != null
              ? { negative_prompt: protect(stage.negative_prompt) }
              : {}),
          })),
        }
      : {}),
  });
}
export function validateClips(
  owner: ChainOwner,
  request: ChainRequestWire,
): Promise<ChainPlan> {
  return apiJsonTo(owner.target, "/api/generate/chain/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: chainBody(request),
  });
}
export async function getChainJob(
  owner: ChainOwner,
  id: string,
  includeSourceMedia = false,
): Promise<ChainJobDetail | null> {
  try {
    return await apiJsonTo(
      owner.target,
      `/api/chain-jobs/${encodeURIComponent(id)}${includeSourceMedia ? "?include_source_media=true" : ""}`,
    );
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
}
/** Persist identity before dispatch. Lost replies reconcile by the SAME job UUID. */
export async function admitClips(
  owner: ChainOwner,
  request: ChainRequestWire,
  persist: (admission: ChainAdmission) => void,
  operationId: string = crypto.randomUUID(),
  isCurrent: () => boolean = () => true,
): Promise<ChainAdmission> {
  await assertChainOwner(owner);
  if (!isCurrent())
    throw new Error("The video draft or machine changed before admission.");
  const admission: ChainAdmission = {
    operationId,
    hostId: owner.hostId,
    instanceId: owner.instanceId!,
    baseUrl: owner.target.baseUrl,
    ...(owner.target.connectionMode
      ? { connectionMode: owner.target.connectionMode }
      : {}),
    createdAt: Date.now(),
  };
  persist(admission); // Storage failure deliberately prevents POST.
  await apiJsonTo(owner.target, "/api/chain-jobs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-mold-operation-id": operationId,
    },
    body: chainBody(request),
  });
  return admission;
}
export async function mutateChain(
  owner: ChainOwner,
  id: string,
  action: "resume" | "cancel",
  isCurrent: () => boolean = () => true,
): Promise<void> {
  await assertChainOwner(owner);
  if (!isCurrent())
    throw new Error("The video or machine changed before this action.");
  await apiJsonTo(
    owner.target,
    `/api/chain-jobs/${encodeURIComponent(id)}/${action}`,
    { method: "POST" },
  );
}

/** Cancels the UUID known before create returns. The 202 body is empty. */
export async function cancelChainOperation(
  owner: ChainOwner,
  id: string,
  isCurrent: () => boolean = () => true,
): Promise<void> {
  await assertChainOwner(owner);
  if (!isCurrent())
    throw new Error("The video or machine changed before cancellation.");
  await apiFetchTo(
    owner.target,
    `/api/chain-jobs/${encodeURIComponent(id)}/operations/${encodeURIComponent(id)}/cancel`,
    { method: "POST" },
  );
}
