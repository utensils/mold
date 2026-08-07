import { apiJsonTo, type ApiTarget } from "./client";
import {
  redactGenerationReference,
  type GenerationReference,
} from "../lib/generationReferences";

export interface GenerationPlacementCandidate {
  device_id: string;
  execution_fingerprint: string;
  predicted_start_after_ms: number;
  predicted_completion_after_ms: number;
  setup_ms: number;
  setup_kind: string;
  estimate_confidence: string;
}

export interface GenerationPlacementPreview {
  version: number;
  authoritative: boolean;
  state_version: number;
  plan_version: number;
  outcome: string;
  reason?: string | null;
  candidate?: GenerationPlacementCandidate | null;
  pending_downloads?: PlacementPendingDownload[];
  missing_components?: PlacementMissingComponent[];
  stage_candidates?: Array<
    GenerationPlacementCandidate & {
      stage_index: number;
      copy_index?: number;
    }
  >;
}

export interface PlacementPendingDownload {
  kind: string;
  name: string;
  repo: string;
  bytes?: number | null;
}

export interface PlacementMissingComponent {
  kind: string;
  name: string;
  present: boolean;
  path?: string | null;
  repair_model?: string | null;
}

export type PlacementPreviewClassification =
  | "planned"
  | "unsupported"
  | "infeasible"
  | "temporarily_unavailable"
  | "invalid";

function nonNegativeSafeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function hasNoCandidates(preview: Record<string, unknown>): boolean {
  return (
    (preview.candidate === undefined || preview.candidate === null) &&
    (preview.stage_candidates === undefined ||
      (Array.isArray(preview.stage_candidates) &&
        preview.stage_candidates.length === 0))
  );
}

function validMissingComponents(value: unknown): boolean {
  if (value === undefined) return true;
  if (!Array.isArray(value)) return false;
  return value.every((component) => {
    if (
      typeof component !== "object" ||
      component === null ||
      Array.isArray(component)
    ) {
      return false;
    }
    const row = component as Record<string, unknown>;
    return (
      typeof row.kind === "string" &&
      row.kind.length > 0 &&
      typeof row.name === "string" &&
      row.name.length > 0 &&
      typeof row.present === "boolean" &&
      (row.path === undefined ||
        row.path === null ||
        typeof row.path === "string") &&
      (row.repair_model === undefined ||
        row.repair_model === null ||
        typeof row.repair_model === "string")
    );
  });
}

function validPendingDownloads(value: unknown): boolean {
  if (value === undefined) return true;
  if (!Array.isArray(value)) return false;
  return value.every((download) => {
    if (
      typeof download !== "object" ||
      download === null ||
      Array.isArray(download)
    ) {
      return false;
    }
    const row = download as Record<string, unknown>;
    return (
      typeof row.kind === "string" &&
      row.kind.length > 0 &&
      typeof row.name === "string" &&
      row.name.length > 0 &&
      typeof row.repo === "string" &&
      row.repo.length > 0 &&
      (row.bytes === undefined ||
        row.bytes === null ||
        nonNegativeSafeInteger(row.bytes))
    );
  });
}

/** Runtime wire validation for placement authority. A malformed response must
 * never silently downgrade to legacy routing. */
export function classifyPlacementPreview(
  value: unknown,
): PlacementPreviewClassification {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    return "invalid";
  const preview = value as Record<string, unknown>;
  if (
    preview.version !== 1 ||
    !nonNegativeSafeInteger(preview.state_version) ||
    !nonNegativeSafeInteger(preview.plan_version)
  ) {
    return "invalid";
  }
  if (
    preview.authoritative === false &&
    preview.outcome === "unsupported" &&
    hasNoCandidates(preview)
  ) {
    return "unsupported";
  }
  if (
    (preview.authoritative === true || preview.authoritative === false) &&
    preview.outcome === "temporarily_unavailable" &&
    typeof preview.reason === "string" &&
    preview.reason.length > 0 &&
    hasNoCandidates(preview)
  ) {
    return "temporarily_unavailable";
  }
  if (
    preview.authoritative === true &&
    preview.outcome === "infeasible" &&
    typeof preview.reason === "string" &&
    preview.reason.length > 0 &&
    hasNoCandidates(preview) &&
    validMissingComponents(preview.missing_components)
  ) {
    return "infeasible";
  }
  if (
    preview.authoritative !== true ||
    preview.outcome !== "planned" ||
    typeof preview.candidate !== "object" ||
    preview.candidate === null ||
    Array.isArray(preview.candidate)
  ) {
    return "invalid";
  }
  const candidate = preview.candidate as Record<string, unknown>;
  if (
    typeof candidate.device_id !== "string" ||
    candidate.device_id.length === 0 ||
    typeof candidate.execution_fingerprint !== "string" ||
    candidate.execution_fingerprint.length === 0 ||
    !nonNegativeSafeInteger(candidate.predicted_start_after_ms) ||
    !nonNegativeSafeInteger(candidate.predicted_completion_after_ms) ||
    candidate.predicted_completion_after_ms <
      candidate.predicted_start_after_ms ||
    !nonNegativeSafeInteger(candidate.setup_ms) ||
    typeof candidate.setup_kind !== "string" ||
    candidate.setup_kind.length === 0 ||
    typeof candidate.estimate_confidence !== "string" ||
    candidate.estimate_confidence.length === 0 ||
    !validPendingDownloads(preview.pending_downloads)
  ) {
    return "invalid";
  }
  return "planned";
}

/** A prepared Batch N is submitted as N independent one-image requests.
 * Keep the reviewed request immutable, but preview the exact sibling shape
 * instead of multiplying its original batch_size by copies a second time. */
export function previewRequestForSiblingFanout<
  T extends Record<string, unknown>,
>(request: T, copies: number): T {
  if (copies <= 1 || request.batch_size === undefined) return request;
  return { ...request, batch_size: 1 } as T;
}

function redactPresent(
  target: Record<string, unknown>,
  source: Record<string, unknown>,
  field: string,
): void {
  if (source[field] !== undefined && source[field] !== null) target[field] = "";
}

/** Preserve every planning-relevant structural field without sending user
 * prompt or media contents to candidate hosts. */
export function redactGenerationForPlacement<T extends Record<string, unknown>>(
  request: T,
): T {
  const redacted: Record<string, unknown> = { ...request, prompt: "" };
  for (const field of [
    "negative_prompt",
    "original_prompt",
    "source_image_name",
    "source_image",
    "mask_image",
    "control_image",
    "audio_file_path",
    "audio_file",
    "source_video_path",
    "source_video",
  ]) {
    redactPresent(redacted, request, field);
  }
  if (Array.isArray(request.edit_images)) {
    redacted.edit_images = request.edit_images.map(() => "");
  }
  if (Array.isArray(request.references)) {
    redacted.references = request.references.map((reference) =>
      typeof reference === "object" &&
      reference !== null &&
      !Array.isArray(reference)
        ? redactGenerationReference(reference as GenerationReference)
        : reference,
    );
  }
  if (Array.isArray(request.keyframes)) {
    redacted.keyframes = request.keyframes.map((value) => {
      if (typeof value !== "object" || value === null || Array.isArray(value))
        return value;
      const keyframe = value as Record<string, unknown>;
      return { ...keyframe, image: "" };
    });
  }
  return redacted as T;
}

export async function previewGenerationPlacement(
  target: ApiTarget,
  request: Record<string, unknown>,
  copies = 1,
): Promise<GenerationPlacementPreview> {
  return apiJsonTo<GenerationPlacementPreview>(
    target,
    "/api/generate/placement-preview",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        request: redactGenerationForPlacement(request),
        copies,
      }),
    },
  );
}

/** Preserve sequence topology and conditioning presence while removing prompt
 * and image contents before probing another host. */
export function redactChainForPlacement<T extends Record<string, unknown>>(
  request: T,
): T {
  const redacted: Record<string, unknown> = { ...request };
  if (redacted.prompt !== undefined && redacted.prompt !== null)
    redacted.prompt = "";
  redactPresent(redacted, request, "original_prompt");
  redactPresent(redacted, request, "source_image");
  if (Array.isArray(request.stages)) {
    redacted.stages = request.stages.map((value) => {
      if (typeof value !== "object" || value === null || Array.isArray(value))
        return value;
      const stage = value as Record<string, unknown>;
      const copy: Record<string, unknown> = { ...stage, prompt: "" };
      redactPresent(copy, stage, "negative_prompt");
      redactPresent(copy, stage, "source_image");
      return copy;
    });
  }
  return redacted as T;
}

export async function previewChainPlacement(
  target: ApiTarget,
  request: Record<string, unknown>,
  copies = 1,
): Promise<GenerationPlacementPreview> {
  return apiJsonTo<GenerationPlacementPreview>(
    target,
    "/api/chain-jobs/placement-preview",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        request: redactChainForPlacement(request),
        copies,
      }),
    },
  );
}

export function comparePlacementPreviews(
  left: {
    hostId: string;
    roundTripMs: number;
    preview: GenerationPlacementPreview;
  },
  right: {
    hostId: string;
    roundTripMs: number;
    preview: GenerationPlacementPreview;
  },
): number {
  const leftPending = (left.preview.pending_downloads?.length ?? 0) > 0;
  const rightPending = (right.preview.pending_downloads?.length ?? 0) > 0;
  if (leftPending !== rightPending) return leftPending ? 1 : -1;
  const leftCandidate = left.preview.candidate!;
  const rightCandidate = right.preview.candidate!;
  return (
    leftCandidate.predicted_completion_after_ms +
      left.roundTripMs -
      (rightCandidate.predicted_completion_after_ms + right.roundTripMs) ||
    leftCandidate.setup_ms - rightCandidate.setup_ms ||
    left.hostId.localeCompare(right.hostId)
  );
}
