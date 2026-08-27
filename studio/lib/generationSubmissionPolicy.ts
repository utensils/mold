import {
  canonicalGenerationBatchLimit,
  type DurableGenerationQueueCapabilities,
  type DurableMediaCapabilities,
  type GenerationBatchChild,
} from "../api/generationAdmission";

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

/**
 * Whether this machine speaks the durable submission contract at all.
 *
 * Deliberately HOST-level and blind to the request: the durable protocol
 * carries source media, LoRAs, identity photos, and H3's ordered references,
 * so a client-side per-trait fence could only ever refuse work the server
 * would have taken. The ONE host-level fact is the durable queue itself.
 * `durable_media` is deliberately not required here: a host whose encrypted
 * media store is degraded still admits every media-free print and refuses a
 * media-carrying one by name (`503 DURABLE_MEDIA_UNAVAILABLE`), and that typed
 * refusal — surfaced through `isDefiniteGenerationAdmissionRejection` — is the
 * single authority for anything it cannot take.
 */
function hostContractRefusal(host: GenerationSubmissionHost): string | null {
  if (canonicalGenerationBatchLimit(host.queue) === null) {
    return "this machine does not advertise the durable generation queue";
  }
  return null;
}

export function generationHostSubmissionPolicy(
  target: GenerationTargetPolicy,
  host: GenerationSubmissionHost,
  outputKind: GenerationSubmissionOutputKind = "generation",
): GenerationHostSubmissionPolicy {
  if (outputKind === "sequence") {
    return {
      routing: "placement_preview",
      admission: "refused",
      refusal: "a sequence is created through the chain-job route",
    };
  }
  const refusal = hostContractRefusal(host);
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
