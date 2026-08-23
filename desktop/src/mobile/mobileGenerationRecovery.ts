import type { GenerateRequest, OutputFormat } from "../lib/api/types";
import type {
  DurableGenerationQueueCapabilities,
  GenerationBatchStatusResponse,
} from "@studio/api/generationAdmission";
import { supportsDurableGenerationLifecycle } from "@studio/api/generationAdmission";
import {
  buildGenerationBatchStatusRequest,
  createGenerationBatchTracker,
  isTerminalGenerationPhase,
  mergeBulkGenerationBatchResponse,
  reduceGenerationLifecycle,
  type GenerationBatchTracker,
  type GenerationLifecycleAction,
  type GenerationLifecycleJob,
} from "@studio/lib/generationLifecycle";
import { isMinimaxH3Identity } from "@studio/lib/minimaxH3Authoring";

export const MOBILE_DURABLE_GENERATIONS_KEY = "mold.mobile.durable-generations.v1";

/**
 * Byte-free presentation state retained across a WebView restart. It is not a
 * generation request: source/identity/control media, prompts, filing, routes,
 * API keys and upload handles are deliberately absent.
 */
export interface MobileDurableGenerationPresentation {
  index: number;
  model: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number;
  format: OutputFormat;
  submittedAtMs: number;
}

export type MobileDurableTerminalEffect = "viewer" | "photos" | "gallery";

export interface MobileDurableGenerationRecovery {
  version: 1;
  tracker: GenerationBatchTracker;
  presentations: MobileDurableGenerationPresentation[];
  /** Effect claims are persisted before side effects run. This makes a wake,
   * reconnect, duplicate event or process restart unable to repeat them. */
  claimedEffects: Record<string, Partial<Record<MobileDurableTerminalEffect, true>>>;
}

export interface MobileDurableHostIdentity {
  id: string;
  instanceId?: string | null | undefined;
}

const OUTPUT_FORMATS = new Set<OutputFormat>(["png", "jpeg", "webp", "gif", "apng", "mp4", "wav"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

type ExtendedMediaGenerateRequest = GenerateRequest & {
  /** Additive server wire fields not yet authored by the shared mobile form. */
  id_images?: readonly string[] | null;
  audio_file_path?: string | null;
  source_video_path?: string | null;
  hdr_exr_dir?: string | null;
};

function parsePresentation(value: unknown): MobileDurableGenerationPresentation | null {
  if (
    !isRecord(value) ||
    !Number.isInteger(value.index) ||
    Number(value.index) < 1 ||
    typeof value.model !== "string" ||
    value.model.length === 0 ||
    !finiteNumber(value.width) ||
    !finiteNumber(value.height) ||
    !finiteNumber(value.steps) ||
    !finiteNumber(value.guidance) ||
    !finiteNumber(value.seed) ||
    typeof value.format !== "string" ||
    !OUTPUT_FORMATS.has(value.format as OutputFormat) ||
    !finiteNumber(value.submittedAtMs)
  ) {
    return null;
  }
  return {
    index: Number(value.index),
    model: value.model,
    width: value.width,
    height: value.height,
    steps: value.steps,
    guidance: value.guidance,
    seed: value.seed,
    format: value.format as OutputFormat,
    submittedAtMs: value.submittedAtMs,
  };
}

function parseTracker(value: unknown): GenerationBatchTracker | null {
  if (
    !isRecord(value) ||
    typeof value.hostId !== "string" ||
    !value.hostId ||
    typeof value.expectedInstanceId !== "string" ||
    !value.expectedInstanceId ||
    typeof value.clientBatchId !== "string" ||
    !value.clientBatchId ||
    (value.serverBatchId !== null && typeof value.serverBatchId !== "string") ||
    !isRecord(value.admission) ||
    !isRecord(value.reconciliation) ||
    !isRecord(value.jobs)
  ) {
    return null;
  }
  // The canonical reducer remains the semantic validator when snapshots are
  // merged. Recovery parsing only accepts the serializable shape and rejects
  // anything capable of smuggling route or request data into this record.
  const allowed = new Set([
    "hostId",
    "expectedInstanceId",
    "clientBatchId",
    "serverBatchId",
    "admission",
    "reconciliation",
    "jobs",
  ]);
  if (Object.keys(value).some((key) => !allowed.has(key))) return null;
  return value as unknown as GenerationBatchTracker;
}

function parseClaimedEffects(
  value: unknown,
): Record<string, Partial<Record<MobileDurableTerminalEffect, true>>> | null {
  if (!isRecord(value)) return null;
  const parsed: Record<string, Partial<Record<MobileDurableTerminalEffect, true>>> = {};
  for (const [key, effects] of Object.entries(value)) {
    if (!key || !isRecord(effects)) return null;
    const next: Partial<Record<MobileDurableTerminalEffect, true>> = {};
    for (const effect of ["viewer", "photos", "gallery"] as const) {
      if (effects[effect] === true) next[effect] = true;
      else if (effects[effect] !== undefined) return null;
    }
    if (Object.keys(effects).some((effect) => !["viewer", "photos", "gallery"].includes(effect))) {
      return null;
    }
    parsed[key] = next;
  }
  return parsed;
}

export function parseMobileDurableGenerationRecovery(
  value: unknown,
): MobileDurableGenerationRecovery | null {
  if (!isRecord(value) || value.version !== 1 || !Array.isArray(value.presentations)) return null;
  if (
    Object.keys(value).some(
      (key) => !["version", "tracker", "presentations", "claimedEffects"].includes(key),
    )
  ) {
    return null;
  }
  const tracker = parseTracker(value.tracker);
  const presentations = value.presentations.map(parsePresentation);
  const claimedEffects = parseClaimedEffects(value.claimedEffects);
  if (!tracker || presentations.some((entry) => entry === null) || !claimedEffects) return null;
  const indexes = new Set<number>();
  for (const presentation of presentations as MobileDurableGenerationPresentation[]) {
    if (indexes.has(presentation.index)) return null;
    indexes.add(presentation.index);
  }
  return {
    version: 1,
    tracker,
    presentations: presentations as MobileDurableGenerationPresentation[],
    claimedEffects,
  };
}

export function loadMobileDurableGenerationRecoveries(
  storage: Pick<Storage, "getItem">,
): MobileDurableGenerationRecovery[] {
  try {
    const raw = JSON.parse(storage.getItem(MOBILE_DURABLE_GENERATIONS_KEY) ?? "[]");
    if (!Array.isArray(raw)) return [];
    return raw.flatMap((entry) => {
      const parsed = parseMobileDurableGenerationRecovery(entry);
      return parsed ? [parsed] : [];
    });
  } catch {
    return [];
  }
}

export function saveMobileDurableGenerationRecoveries(
  storage: Pick<Storage, "setItem">,
  records: readonly MobileDurableGenerationRecovery[],
): void {
  storage.setItem(MOBILE_DURABLE_GENERATIONS_KEY, JSON.stringify(records));
}

export function generationRequestCarriesMedia(request: GenerateRequest): boolean {
  const extended = request as ExtendedMediaGenerateRequest;
  return (
    request.source_image != null ||
    request.id_image != null ||
    request.mask_image != null ||
    request.control_image != null ||
    request.audio_file != null ||
    request.source_video != null ||
    request.extend_video != null ||
    request.extend_video_path != null ||
    request.edit_images !== undefined ||
    request.keyframes !== undefined ||
    request.references !== undefined ||
    extended.id_images != null ||
    extended.audio_file_path != null ||
    extended.source_video_path != null ||
    extended.hdr_exr_dir != null
  );
}

/** Durable admission is intentionally narrower than server capability: media
 * requests stay on the authenticated legacy stream until their bytes have a
 * durable, encrypted staging authority. Automatic chains keep their existing
 * durable sequence protocol. */
export function useMobileDurableGenerationLifecycle(input: {
  queue: DurableGenerationQueueCapabilities | null | undefined;
  requests: readonly GenerateRequest[];
  chain: boolean;
  modelFamily?: string | null;
}): boolean {
  return (
    !input.chain &&
    supportsDurableGenerationLifecycle(input.queue) &&
    input.requests.length > 0 &&
    !input.requests.some((request) => isMinimaxH3Identity(input.modelFamily, request.model)) &&
    !input.requests.some(generationRequestCarriesMedia)
  );
}

export function mobileDurablePresentations(
  requests: readonly GenerateRequest[],
  submittedAtMs: number,
): MobileDurableGenerationPresentation[] {
  return requests.map((request, offset) => ({
    index: offset + 1,
    model: request.model,
    width: request.width,
    height: request.height,
    steps: request.steps,
    guidance: request.guidance ?? 1,
    seed: request.seed ?? 0,
    format: request.output_format ?? "png",
    submittedAtMs,
  }));
}

export function createMobileDurableGenerationRecovery(input: {
  hostId: string;
  expectedInstanceId: string;
  clientBatchId: string;
  requests: readonly GenerateRequest[];
  submittedAtMs: number;
}): MobileDurableGenerationRecovery {
  return {
    version: 1,
    tracker: createGenerationBatchTracker({
      hostId: input.hostId,
      expectedInstanceId: input.expectedInstanceId,
      clientBatchId: input.clientBatchId,
      submittedAtMs: input.submittedAtMs,
    }),
    presentations: mobileDurablePresentations(input.requests, input.submittedAtMs),
    claimedEffects: {},
  };
}

export function reduceMobileDurableGenerationRecovery(
  recovery: MobileDurableGenerationRecovery,
  action: GenerationLifecycleAction,
): MobileDurableGenerationRecovery {
  const tracker = reduceGenerationLifecycle(recovery.tracker, action);
  return tracker === recovery.tracker ? recovery : { ...recovery, tracker };
}

export function buildMobileDurableHostStatusRequest(
  records: readonly MobileDurableGenerationRecovery[],
  hostId: string,
) {
  return buildGenerationBatchStatusRequest(
    records.map((record) => record.tracker),
    hostId,
  );
}

export function mergeMobileDurableHostStatus(
  records: readonly MobileDurableGenerationRecovery[],
  hostId: string,
  response: GenerationBatchStatusResponse,
): MobileDurableGenerationRecovery[] {
  const merged = mergeBulkGenerationBatchResponse(
    records.map((record) => record.tracker),
    hostId,
    response,
  ).trackers;
  return records.map((record, index) =>
    merged[index] === record.tracker ? record : { ...record, tracker: merged[index]! },
  );
}

export function mobileDurableJobs(
  recovery: MobileDurableGenerationRecovery,
): GenerationLifecycleJob[] {
  return Object.values(recovery.tracker.jobs).sort(
    (left, right) => left.childIndex - right.childIndex,
  );
}

export function mobileDurableRecoveryIsTerminal(
  recovery: MobileDurableGenerationRecovery,
): boolean {
  const jobs = mobileDurableJobs(recovery);
  return (
    recovery.tracker.admission.phase === "rejected" ||
    (jobs.length === recovery.presentations.length &&
      jobs.length > 0 &&
      jobs.every((job) => isTerminalGenerationPhase(job.phase)))
  );
}

export function mobileDurableAdmissionEffectKey(recovery: MobileDurableGenerationRecovery): string {
  return `admission:${recovery.tracker.clientBatchId}`;
}

/** A terminal recovery may leave durable storage only after every effect has
 * been claimed. Active recoveries never enter this path, so queued work remains
 * unlimited while completed history cannot consume the storage quota forever. */
export function mobileDurableTerminalEffectsClaimed(
  recovery: MobileDurableGenerationRecovery,
): boolean {
  if (!mobileDurableRecoveryIsTerminal(recovery)) return false;
  if (recovery.tracker.admission.phase === "rejected") {
    return recovery.claimedEffects[mobileDurableAdmissionEffectKey(recovery)]?.viewer === true;
  }
  return mobileDurableJobs(recovery).every((job) => {
    const effects = recovery.claimedEffects[job.key];
    return (
      effects?.viewer === true &&
      (job.phase !== "complete" || (effects.photos === true && effects.gallery === true))
    );
  });
}

export function claimMobileDurableTerminalEffect(
  recovery: MobileDurableGenerationRecovery,
  authorityKey: string,
  effect: MobileDurableTerminalEffect,
): { recovery: MobileDurableGenerationRecovery; claimed: boolean } {
  const current = recovery.claimedEffects[authorityKey];
  if (current?.[effect]) return { recovery, claimed: false };
  return {
    claimed: true,
    recovery: {
      ...recovery,
      claimedEffects: {
        ...recovery.claimedEffects,
        [authorityKey]: { ...current, [effect]: true },
      },
    },
  };
}

/** Resolve native credentials only from the current host store and only while
 * the stable host id and exact server instance still match. */
export function resolveMobileDurableHost<T extends MobileDurableHostIdentity>(
  recovery: MobileDurableGenerationRecovery,
  hosts: readonly T[],
): T | null {
  return (
    hosts.find(
      (host) =>
        host.id === recovery.tracker.hostId &&
        (host.instanceId?.trim() ?? "") === recovery.tracker.expectedInstanceId,
    ) ?? null
  );
}
