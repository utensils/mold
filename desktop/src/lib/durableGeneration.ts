import type { GenerateRequest, OutputFormat } from "./api/types";
import type { GenerationBatchTracker } from "@studio/lib/generationLifecycle";
import {
  type DurableGenerationQueueCapabilities,
  type DurableMediaCapabilities,
} from "@studio/api/generationAdmission";
import { generationHostSubmissionPolicy } from "@studio/lib/generationSubmissionPolicy";

export const DURABLE_GENERATION_STORAGE_KEY = "mold.desktop.durable-generations.v1";

export interface DurableGenerationChildSummary {
  index: number;
  clientId: number | null;
  model: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number | null;
  format: OutputFormat;
}

export interface DurableGenerationRecoveryRecord {
  tracker: GenerationBatchTracker;
  hostLabel: string;
  hostKind: "local" | "remote";
  mirrorRemoteOutput: boolean;
  children: DurableGenerationChildSummary[];
  /** Child indexes whose user cancellation must be completed once the
   * durable admission exposes the server-owned queue id. Persisted before
   * any lookup/DELETE so one tap survives a lost response or app restart. */
  cancelRequestedChildIndexes: number[];
  /** At-most-once receipts. They are persisted before performing effects. */
  effectReceipts: string[];
}

export interface DurableGenerationRecoveryEnvelope {
  version: 1;
  records: DurableGenerationRecoveryRecord[];
}

/** Decide against the exact frozen host capability through the one shared
 * submission policy. A request this machine cannot carry durably is refused,
 * never re-routed to a second submission path. */
export function requestIsEligibleForDurableGeneration(
  request: GenerateRequest,
  queue: DurableGenerationQueueCapabilities | null | undefined,
  durableMedia: DurableMediaCapabilities | null | undefined,
  modelFamily?: string | null,
): boolean {
  return (
    generationHostSubmissionPolicy(
      { kind: "pinned", hostId: "frozen" },
      {
        hostId: "frozen",
        ...(queue === undefined ? {} : { queue }),
        ...(durableMedia === undefined ? {} : { durableMedia }),
      },
      modelFamily ? { ...request, family: modelFamily } : request,
    ).admission === "canonical_durable"
  );
}

export function durableChildSummary(
  request: GenerateRequest,
  index: number,
  clientId: number | null,
): DurableGenerationChildSummary {
  return {
    index,
    clientId,
    model: request.model,
    width: request.width,
    height: request.height,
    steps: request.steps,
    guidance: request.guidance ?? 1,
    seed: request.seed ?? null,
    format: request.output_format ?? "png",
  };
}

export function loadDurableGenerationRecovery(
  storage: Pick<Storage, "getItem"> | null = typeof localStorage === "undefined"
    ? null
    : localStorage,
): DurableGenerationRecoveryRecord[] {
  if (!storage) return [];
  try {
    const parsed = JSON.parse(
      storage.getItem(DURABLE_GENERATION_STORAGE_KEY) ?? "null",
    ) as DurableGenerationRecoveryEnvelope | null;
    if (parsed?.version !== 1 || !Array.isArray(parsed.records)) return [];
    return parsed.records.flatMap((record) => {
      if (
        !record ||
        typeof record !== "object" ||
        typeof record.tracker?.clientBatchId !== "string" ||
        typeof record.tracker?.hostId !== "string" ||
        typeof record.tracker?.expectedInstanceId !== "string" ||
        !Array.isArray(record.children) ||
        !Array.isArray(record.effectReceipts)
      ) {
        return [];
      }
      const cancelRequestedChildIndexes = Array.isArray(record.cancelRequestedChildIndexes)
        ? record.cancelRequestedChildIndexes.filter(
            (index) => Number.isSafeInteger(index) && index > 0,
          )
        : [];
      return [{ ...record, cancelRequestedChildIndexes }];
    });
  } catch {
    return [];
  }
}

export function saveDurableGenerationRecovery(
  records: Iterable<DurableGenerationRecoveryRecord>,
  storage: Pick<Storage, "setItem"> | null = typeof localStorage === "undefined"
    ? null
    : localStorage,
): boolean {
  if (!storage) return false;
  const envelope: DurableGenerationRecoveryEnvelope = {
    version: 1,
    records: [...records],
  };
  try {
    storage.setItem(DURABLE_GENERATION_STORAGE_KEY, JSON.stringify(envelope));
    return true;
  } catch {
    // Web Storage can synchronously reject writes in privacy mode or when its
    // quota is exhausted. Recovery then degrades to this process's in-memory
    // UUID/instance tracker; it must never become permission to skip durable
    // server admission or fall back to a second, legacy submission.
    return false;
  }
}

export function parseEventAuthority(data: string): { instanceId: string } | null {
  try {
    const value = JSON.parse(data) as { instance_id?: unknown };
    return typeof value.instance_id === "string" && value.instance_id.length > 0
      ? { instanceId: value.instance_id }
      : null;
  } catch {
    return null;
  }
}

export function parseEventResync(data: string): {
  instanceId: string;
  missedEvents: number;
} | null {
  try {
    const value = JSON.parse(data) as {
      instance_id?: unknown;
      missed_events?: unknown;
    };
    return typeof value.instance_id === "string" &&
      value.instance_id.length > 0 &&
      typeof value.missed_events === "number" &&
      Number.isSafeInteger(value.missed_events) &&
      value.missed_events >= 0
      ? { instanceId: value.instance_id, missedEvents: value.missed_events }
      : null;
  } catch {
    return null;
  }
}
