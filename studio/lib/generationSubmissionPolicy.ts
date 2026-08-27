import {
  canonicalGenerationBatchLimit,
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

/**
 * `placement_preview` belongs to sequences alone — a chain job is planned
 * before it is created. A generation is admitted durably and is never
 * previewed for placement; `telemetry_only` is the read-only probe an
 * automatic target fans out purely to rank machines.
 */
export type GenerationRoutingMode =
  "none" | "telemetry_only" | "placement_preview";

/**
 * A generation is admitted through `POST /api/generation-batches` or it is
 * refused by name. There is no attached-stream transport to fall back to.
 */
export type GenerationAdmissionTransport = "canonical_durable" | "refused";

export interface GenerationHostSubmissionPolicy {
  routing: GenerationRoutingMode;
  admission: GenerationAdmissionTransport;
  /** Named, user-facing reason whenever `admission` is "refused". */
  refusal: string | null;
}

export type GenerationSubmissionOutputKind = "generation" | "sequence";

function fieldPresent(
  request: Record<string, unknown>,
  field: string,
): boolean {
  return request[field] !== undefined && request[field] !== null;
}

/**
 * Durable media is one complete replay contract. Request traits select
 * independently advertised guarantees; H3 additionally carries an explicit
 * private durable contract even when the particular request is media-free.
 * A trait the host does not cover is REFUSED by name — never routed to a
 * second submission path. The server remains authoritative for model
 * preparation and for every execution check.
 */
function generationRefusal(
  host: GenerationSubmissionHost,
  request: object,
): string | null {
  if (canonicalGenerationBatchLimit(host.queue) === null) {
    return "this machine does not advertise the durable generation queue";
  }
  const record = request as Record<string, unknown>;
  if (fieldPresent(record, "hdr_exr_dir")) {
    return "an HDR EXR output directory cannot be queued";
  }
  const carriesMedia = requestCarriesGenerationMedia(request);
  if (
    carriesMedia &&
    (fieldPresent(record, "lora") || fieldPresent(record, "loras"))
  ) {
    return "a LoRA cannot be combined with source media in a queued print";
  }
  const h3 = isMinimaxH3Identity(
    typeof record.family === "string" ? record.family : null,
    typeof record.model === "string" ? record.model : null,
  );
  if (!carriesMedia && !h3) return null;
  const media = host.durableMedia;
  if (
    !media ||
    media.encrypted_at_rest !== true ||
    media.generate_request_media !== true
  ) {
    return "this machine cannot store request media durably";
  }
  if (h3 && media.private_h3 !== true) {
    return "this machine cannot store MiniMax H3 request media durably";
  }
  if (
    (fieldPresent(record, "id_image") || fieldPresent(record, "id_images")) &&
    media.identity !== true
  ) {
    return "this machine cannot store identity photos durably";
  }
  if (fieldPresent(record, "references") && media.h3_references !== true) {
    return "this machine cannot store reference media durably";
  }
  return null;
}

export function generationHostSubmissionPolicy(
  target: GenerationTargetPolicy,
  host: GenerationSubmissionHost,
  request: object,
  outputKind: GenerationSubmissionOutputKind = "generation",
): GenerationHostSubmissionPolicy {
  if (outputKind === "sequence") {
    return {
      routing: "placement_preview",
      admission: "refused",
      refusal: "a sequence is created through the chain-job route",
    };
  }
  const refusal = generationRefusal(host, request);
  if (refusal !== null) {
    return { routing: "none", admission: "refused", refusal };
  }
  return {
    routing: target.kind === "pinned" ? "none" : "telemetry_only",
    admission: "canonical_durable",
    refusal: null,
  };
}

export type GenerationTruthfulPhase =
  "accepted" | "held" | "queued" | "running" | "cancelling" | "terminal";

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
  if (state === "cancelling") return "cancelling";
  if (state === "accepted") return "accepted";
  return "terminal";
}
