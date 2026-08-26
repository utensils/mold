import {
  canonicalGenerationBatchLimit,
  isDurableMediaCapabilitiesV1,
  supportsDurableGenerationLifecycle,
  supportsDurableRequest,
  type DurableGenerationQueueCapabilities,
  type DurableMediaCapabilities,
  type GenerationBatchChild,
} from "../api/generationAdmission";
import { requestCarriesGenerationMedia } from "./generationMedia";
import { isMinimaxH3Identity } from "./minimaxH3Identity";

export type GenerationTargetPolicy =
  { kind: "pinned"; hostId: string } | { kind: "auto" } | { kind: "capable" };

export interface GenerationSubmissionHost {
  hostId: string;
  queue?: DurableGenerationQueueCapabilities | null;
  durableMedia?: DurableMediaCapabilities | null;
}

export type GenerationRoutingMode =
  "none" | "telemetry_only" | "legacy_placement";

export type GenerationAdmissionTransport =
  "canonical_durable" | "legacy_durable" | "legacy_attached";

export interface GenerationHostSubmissionPolicy {
  compatibility: "canonical_v2" | "legacy";
  routing: GenerationRoutingMode;
  admission: GenerationAdmissionTransport;
}

export interface GenerationSubmissionPlan {
  target: GenerationTargetPolicy;
  hosts: Array<GenerationSubmissionHost & GenerationHostSubmissionPolicy>;
}

export type GenerationSubmissionOutputKind = "generation" | "sequence";

function fieldPresent(
  request: Record<string, unknown>,
  field: string,
): boolean {
  return request[field] !== undefined && request[field] !== null;
}

/**
 * Protocol-v2 media is one complete replay contract. Request traits select
 * independently versioned guarantees; H3 additionally carries an explicit
 * private durable contract even when the particular request is media-free.
 * The server remains authoritative for model preparation.
 */
function supportsCanonicalRequest(
  host: GenerationSubmissionHost,
  request: object,
): boolean {
  if (
    !supportsDurableGenerationLifecycle(host.queue) ||
    canonicalGenerationBatchLimit(host.queue) === null ||
    !Number.isSafeInteger(host.queue?.admission_protocol_version) ||
    (host.queue?.admission_protocol_version ?? 0) < 2
  ) {
    return false;
  }
  const record = request as Record<string, unknown>;
  const h3 = isMinimaxH3Identity(
    typeof record.family === "string" ? record.family : null,
    typeof record.model === "string" ? record.model : null,
  );
  const carriesMedia = requestCarriesGenerationMedia(request);
  if (fieldPresent(record, "hdr_exr_dir")) return false;
  if (
    carriesMedia &&
    (fieldPresent(record, "lora") || fieldPresent(record, "loras"))
  ) {
    return false;
  }
  if (!carriesMedia && !h3) return true;
  const media = host.durableMedia;
  if (
    !media ||
    !Number.isSafeInteger(media.protocol_version) ||
    media.protocol_version < 2 ||
    media.encrypted_at_rest !== true ||
    media.generate_request_media !== true
  ) {
    return false;
  }
  if (h3 && media.private_h3 !== true) return false;
  if (
    (fieldPresent(record, "id_image") || fieldPresent(record, "id_images")) &&
    media.identity !== true
  ) {
    return false;
  }
  if (fieldPresent(record, "references") && media.h3_references !== true) {
    return false;
  }
  return true;
}

function legacyAdmission(
  host: GenerationSubmissionHost,
  request: object,
): GenerationAdmissionTransport {
  const record = request as Record<string, unknown>;
  if (
    isMinimaxH3Identity(
      typeof record.family === "string" ? record.family : null,
      typeof record.model === "string" ? record.model : null,
    )
  ) {
    return "legacy_attached";
  }
  // Protocol v1 is intentionally exact. An unknown future media protocol must
  // not accidentally activate the legacy replay implementation.
  const durableMedia = isDurableMediaCapabilitiesV1(host.durableMedia)
    ? host.durableMedia
    : undefined;
  return supportsDurableRequest(host.queue, durableMedia, request)
    ? "legacy_durable"
    : "legacy_attached";
}

export function generationHostSubmissionPolicy(
  target: GenerationTargetPolicy,
  host: GenerationSubmissionHost,
  request: object,
  outputKind: GenerationSubmissionOutputKind = "generation",
): GenerationHostSubmissionPolicy {
  if (outputKind === "generation" && supportsCanonicalRequest(host, request)) {
    return {
      compatibility: "canonical_v2",
      routing: target.kind === "pinned" ? "none" : "telemetry_only",
      admission: "canonical_durable",
    };
  }
  return {
    compatibility: "legacy",
    routing: "legacy_placement",
    admission: legacyAdmission(host, request),
  };
}

/** One pure decision point shared by desktop, web, and phone orchestrators. */
export function planGenerationSubmission(input: {
  target: GenerationTargetPolicy;
  hosts: readonly GenerationSubmissionHost[];
  request: object;
  outputKind?: GenerationSubmissionOutputKind;
}): GenerationSubmissionPlan {
  const pinnedHostId =
    input.target.kind === "pinned" ? input.target.hostId : null;
  const hosts =
    pinnedHostId === null
      ? [...input.hosts]
      : input.hosts.filter((host) => host.hostId === pinnedHostId);
  return {
    target: input.target,
    hosts: hosts.map((host) => ({
      ...host,
      ...generationHostSubmissionPolicy(
        input.target,
        host,
        input.request,
        input.outputKind,
      ),
    })),
  };
}

export type GenerationTruthfulPhase =
  "accepted" | "held" | "queued" | "running" | "terminal";

type GenerationPhaseSource =
  | Pick<GenerationBatchChild, "state">
  | { phase: GenerationBatchChild["state"] };

/** Present the authoritative durable-child lifecycle across wire and tracker shapes. */
export function truthfulGenerationPhase(
  child: GenerationPhaseSource,
): GenerationTruthfulPhase {
  const state = "state" in child ? child.state : child.phase;
  if (state === "held") return "held";
  if (state === "queued") return "queued";
  if (state === "running") return "running";
  if (state === "accepted") return "accepted";
  return "terminal";
}
