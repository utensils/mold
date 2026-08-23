import type { GenerateRequest, OutputFormat } from "./api/types";
import type { GenerationBatchTracker } from "@studio/lib/generationLifecycle";
import { requestCarriesGenerationMedia } from "@studio/lib/generationMedia";

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
  /** At-most-once receipts. They are persisted before performing effects. */
  effectReceipts: string[];
}

export interface DurableGenerationRecoveryEnvelope {
  version: 1;
  records: DurableGenerationRecoveryRecord[];
}

/**
 * The current durable journal deliberately excludes temporary/private media
 * authority. Keep those shapes on the legacy attached stream until a server
 * capability explicitly promises encrypted durable media staging.
 */
export function requestIsEligibleForDurableGeneration(request: GenerateRequest): boolean {
  if (request.model.toLowerCase().includes("minimax-h3")) return false;
  return !requestCarriesGenerationMedia(request);
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
    return parsed.records.filter(
      (record) =>
        record &&
        typeof record === "object" &&
        typeof record.tracker?.clientBatchId === "string" &&
        typeof record.tracker?.hostId === "string" &&
        typeof record.tracker?.expectedInstanceId === "string" &&
        Array.isArray(record.children) &&
        Array.isArray(record.effectReceipts),
    );
  } catch {
    return [];
  }
}

export function saveDurableGenerationRecovery(
  records: Iterable<DurableGenerationRecoveryRecord>,
  storage: Pick<Storage, "setItem"> | null = typeof localStorage === "undefined"
    ? null
    : localStorage,
): void {
  if (!storage) return;
  const envelope: DurableGenerationRecoveryEnvelope = {
    version: 1,
    records: [...records],
  };
  storage.setItem(DURABLE_GENERATION_STORAGE_KEY, JSON.stringify(envelope));
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
