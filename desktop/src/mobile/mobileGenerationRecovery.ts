import type { GenerateRequest, OutputFormat } from "../lib/api/types";
import type {
  DurableGenerationQueueCapabilities,
  DurableMediaCapabilities,
  GenerationBatchStatusResponse,
} from "@studio/api/generationAdmission";
import { generationHostSubmissionPolicy } from "@studio/lib/generationSubmissionPolicy";
import { generationTrackerSettled } from "@studio/lib/generationPresentation";
import {
  buildGenerationBatchStatusRequest,
  createGenerationBatchTracker,
  mergeBulkGenerationBatchResponse,
  reduceGenerationLifecycle,
  type GenerationBatchTracker,
  type GenerationLifecycleAction,
  type GenerationLifecycleJob,
} from "@studio/lib/generationLifecycle";

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
  /** Persisted user intent for children cancelled before admission returned
   * their server ids. It contains no route, credential, request, or media. */
  cancelRequestedChildIndexes: number[];
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
      (key) =>
        ![
          "version",
          "tracker",
          "presentations",
          "cancelRequestedChildIndexes",
          "claimedEffects",
        ].includes(key),
    )
  ) {
    return null;
  }
  const tracker = parseTracker(value.tracker);
  const presentations = value.presentations.map(parsePresentation);
  const claimedEffects = parseClaimedEffects(value.claimedEffects);
  if (!tracker || presentations.some((entry) => entry === null) || !claimedEffects) return null;
  const cancelRequestedChildIndexes = Array.isArray(value.cancelRequestedChildIndexes)
    ? value.cancelRequestedChildIndexes.filter(
        (index): index is number => Number.isSafeInteger(index) && Number(index) > 0,
      )
    : [];
  const indexes = new Set<number>();
  for (const presentation of presentations as MobileDurableGenerationPresentation[]) {
    if (indexes.has(presentation.index)) return null;
    indexes.add(presentation.index);
  }
  return {
    version: 1,
    tracker,
    presentations: presentations as MobileDurableGenerationPresentation[],
    cancelRequestedChildIndexes,
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
): boolean {
  try {
    storage.setItem(MOBILE_DURABLE_GENERATIONS_KEY, JSON.stringify(records));
    return true;
  } catch {
    // A WebView may reject localStorage in privacy mode or at quota. Keep the
    // live, byte-free authority record in memory so this session can still
    // reconcile by UUID; storage availability cannot veto server admission.
    return false;
  }
}

/**
 * The named reason this machine cannot be queued to, or `null` when it can.
 * Host-level by construction: the durable protocol carries every request
 * trait, so the server's typed admission refusal is the only authority for
 * what it cannot take.
 */
export function mobileDurableGenerationRefusal(input: {
  queue: DurableGenerationQueueCapabilities | null | undefined;
  durableMedia?: DurableMediaCapabilities | null | undefined;
  hostLabel: string;
  instanceId?: string | null | undefined;
}): string | null {
  if (!input.instanceId?.trim()) {
    return `${input.hostLabel} has not reported its server instance yet. Nothing was queued.`;
  }
  const policy = generationHostSubmissionPolicy(
    { kind: "pinned", hostId: "frozen" },
    {
      hostId: "frozen",
      queue: input.queue ?? null,
      durableMedia: input.durableMedia ?? null,
    },
  );
  return policy.admission === "canonical_durable"
    ? null
    : `${input.hostLabel} cannot queue this print: ${policy.refusal}. Nothing was queued.`;
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
    cancelRequestedChildIndexes: [],
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
  return generationTrackerSettled(recovery.tracker, recovery.presentations.length);
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
    // Only a completion that PUBLISHED a file owes Photos and Library
    // effects; one that named no file settled as a failure.
    const published = job.phase === "complete" && !!job.result?.filename;
    return (
      effects?.viewer === true &&
      (!published || (effects.photos === true && effects.gallery === true))
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
