import type { HostRoute } from "../stores/hosts";
import { resolveStyleId, stylePresetLabel } from "./stylePresets";
import { createUuid } from "@studio/lib/id";
import type { ExpandTask } from "@studio/lib/expandTask";
import type { RemixDimension, RemixSourceKind } from "@studio/lib/promptTransform";
import type { PromptTransformProvenance } from "./api/types";

export type HostSelectionPolicy = string | null;

export interface PreparedExpansionInputs {
  kind?: "expand" | "remix";
  sourcePrompt: string;
  rootPrompt?: string;
  sourceKind?: RemixSourceKind;
  dimensions?: readonly RemixDimension[];
  conditioningFingerprint?: string;
  model: string;
  family: string;
  /** Frozen conditioning policy used to create the reviewed rewrite. */
  task: ExpandTask;
  requestedCount: number;
  /** Style-preset chip active when the expansion was requested (`null` = none).
   * Frozen with the batch — prepared work keeps the chip as its indicator. */
  stylePreset: string | null;
  selectedHostPolicy: HostSelectionPolicy;
}

export interface PreparedExpansionPrompt {
  id: string;
  text: string;
  dimensions?: readonly RemixDimension[];
}

export interface PreparedExpansionBatch extends PreparedExpansionInputs {
  batchId: string;
  /** The frozen GENERATION authority — where every sibling is submitted. */
  route: HostRoute;
  /** Recorded only when expansion ran somewhere else (the generation host
   * lacked the expand model). Provenance and the retry target; it is never
   * where the print is queued. */
  expansionRoute?: HostRoute;
  prompts: PreparedExpansionPrompt[];
}

export interface QuickExpansionSnapshot {
  requestToken: number;
  originalPrompt: string;
  expandedPrompt: string;
  model: string;
  family: string;
  task: ExpandTask;
  /** Chip active when the quick expansion was requested — the bake-and-clear
   * apply clears the live chip, and undo restores it from here. */
  stylePreset: string | null;
  selectedHostPolicy: HostSelectionPolicy;
  /** The frozen GENERATION authority — where the print is submitted. */
  route: HostRoute;
  /** Recorded only when expansion ran on a different machine. */
  expansionRoute?: HostRoute;
  promptTransform?: PromptTransformProvenance;
}

export interface CurrentQuickExpansionInputs {
  expandedPrompt: string;
  model: string;
  family: string;
  task: ExpandTask;
  selectedHostPolicy: HostSelectionPolicy;
  readyHostIds: ReadonlySet<string>;
  hostLabels: ReadonlyMap<string, string>;
  modelLabels?: ReadonlyMap<string, string>;
  hostTargets?: CurrentPreparedExpansionInputs["hostTargets"];
}

export interface CurrentPreparedExpansionInputs extends PreparedExpansionInputs {
  readyHostIds: ReadonlySet<string>;
  hostLabels: ReadonlyMap<string, string>;
  modelLabels?: ReadonlyMap<string, string>;
  hostTargets?: ReadonlyMap<
    string,
    {
      baseUrl: string;
      apiKey: string | null;
      kind: HostRoute["kind"];
      instanceId?: string | null;
    }
  >;
}

function modelLabel(name: string, labels?: ReadonlyMap<string, string>): string {
  return labels?.get(name) ?? name;
}

/**
 * Validate the expansion response as one indivisible batch. Whitespace is
 * normalized only after the response has proven it contains exactly the
 * requested number of non-empty prompts. A malformed response never changes
 * the requested batch size on the user's behalf.
 */
export function validateExpandedPrompts(prompts: readonly string[], expected: number): string[] {
  if (prompts.length !== expected) {
    throw new Error(
      `Expected exactly ${expected} non-empty prompts, but the host returned ${prompts.length}.`,
    );
  }
  const normalized = prompts.map((prompt) => {
    const trimmed = prompt.trim();
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (Array.isArray(parsed) && parsed.length === 1 && typeof parsed[0] === "string") {
        return parsed[0].trim();
      }
    } catch {
      // Ordinary prompt text is not JSON and needs only edge trimming.
    }
    return trimmed;
  });
  const emptyIndex = normalized.findIndex((prompt) => !prompt);
  if (emptyIndex >= 0) {
    throw new Error(
      `Prompt ${emptyIndex + 1} was empty. Expected exactly ${expected} non-empty prompts.`,
    );
  }
  return normalized;
}

export function createPreparedExpansionBatch(
  inputs: PreparedExpansionInputs,
  route: HostRoute,
  prompts: readonly string[],
  requestToken: number,
  batchId = createDurableBatchId(),
): PreparedExpansionBatch {
  return {
    ...inputs,
    batchId,
    route: {
      ...route,
      target: { ...route.target },
    },
    prompts: prompts.map((text, index) => ({
      id: `prepared-${requestToken}-${index + 1}`,
      text,
    })),
  };
}

function createDurableBatchId(): string {
  return createUuid();
}

export function hostSelectionLabel(
  policy: HostSelectionPolicy,
  hostLabels: ReadonlyMap<string, string> = new Map(),
): string {
  if (policy === null) return "Auto";
  if (policy === "capable") return "Most capable";
  return hostLabels.get(policy) ?? policy;
}

/** Legacy chip ids (e.g. "photographic") equal their canonical twin. */
function canonicalStyle(id: string | null): string | null {
  return id ? resolveStyleId(id) : null;
}

/** "Style changed from X to Y." plus removal/addition variants; empty when equal. */
function styleStaleReason(frozen: string | null, current: string | null): string | null {
  const frozenStyle = canonicalStyle(frozen);
  const currentStyle = canonicalStyle(current);
  if (frozenStyle === currentStyle) return null;
  if (frozenStyle && currentStyle) {
    return `Style changed from ${stylePresetLabel(frozenStyle)} to ${stylePresetLabel(currentStyle)}.`;
  }
  if (frozenStyle) {
    return `Style ${stylePresetLabel(frozenStyle)} was removed after these variations were prepared.`;
  }
  if (currentStyle) {
    return `Style ${stylePresetLabel(currentStyle)} was added after these variations were prepared.`;
  }
  return null;
}

function knownInstanceIdsDiffer(
  frozen: string | null | undefined,
  current: string | null | undefined,
): boolean {
  return frozen != null && current != null && frozen !== current;
}

/** Specific, stable reasons why reviewed work no longer matches the form. */
export function preparedExpansionStaleReasons(
  batch: PreparedExpansionBatch,
  current: CurrentPreparedExpansionInputs,
): string[] {
  const reasons: string[] = [];
  if (current.sourcePrompt !== batch.sourcePrompt) {
    reasons.push("Source prompt changed after these variations were prepared.");
  }
  if (current.model !== batch.model) {
    reasons.push(
      `Model changed from "${modelLabel(batch.model, current.modelLabels)}" to "${modelLabel(current.model, current.modelLabels)}".`,
    );
  }
  if (current.family !== batch.family) {
    reasons.push(`Model family changed from "${batch.family}" to "${current.family}".`);
  }
  if (current.task !== batch.task) {
    reasons.push(`Conditioning changed from ${batch.task} to ${current.task}.`);
  }
  if (
    batch.conditioningFingerprint !== undefined &&
    current.conditioningFingerprint !== batch.conditioningFingerprint
  ) {
    reasons.push("Conditioning media changed after these variations were prepared.");
  }
  if (
    batch.kind === "remix" &&
    JSON.stringify(current.dimensions ?? []) !== JSON.stringify(batch.dimensions ?? [])
  ) {
    reasons.push("Remix dimensions changed after these variations were prepared.");
  }
  const styleReason = styleStaleReason(batch.stylePreset, current.stylePreset);
  if (styleReason) reasons.push(styleReason);
  if (current.requestedCount !== batch.requestedCount) {
    reasons.push(`Batch changed from ${batch.requestedCount} to ${current.requestedCount}.`);
  }
  if (current.selectedHostPolicy !== batch.selectedHostPolicy) {
    reasons.push(
      `Host selection changed from ${hostSelectionLabel(batch.selectedHostPolicy, current.hostLabels)} to ${hostSelectionLabel(current.selectedHostPolicy, current.hostLabels)}.`,
    );
  }
  if (!current.readyHostIds.has(batch.route.hostId)) {
    reasons.push(`${batch.route.label} is no longer reachable.`);
  } else {
    const currentTarget = current.hostTargets?.get(batch.route.hostId);
    if (
      currentTarget &&
      (currentTarget.baseUrl !== batch.route.target.baseUrl ||
        currentTarget.apiKey !== batch.route.target.apiKey ||
        currentTarget.kind !== batch.route.kind ||
        knownInstanceIdsDiffer(batch.route.instanceId, currentTarget.instanceId))
    ) {
      reasons.push(`${batch.route.label}'s connection details changed.`);
    }
  }
  return reasons;
}

/** Batch 1 gets the same frozen-route guarantees without showing a review workspace. */
export function quickExpansionStaleReasons(
  snapshot: QuickExpansionSnapshot,
  current: CurrentQuickExpansionInputs,
): string[] {
  const reasons: string[] = [];
  if (current.expandedPrompt !== snapshot.expandedPrompt) {
    reasons.push("Expanded prompt changed after it was prepared.");
  }
  if (current.model !== snapshot.model) {
    reasons.push(
      `Model changed from "${modelLabel(snapshot.model, current.modelLabels)}" to "${modelLabel(current.model, current.modelLabels)}".`,
    );
  }
  if (current.family !== snapshot.family) {
    reasons.push(`Model family changed from "${snapshot.family}" to "${current.family}".`);
  }
  if (current.task !== snapshot.task) {
    reasons.push(`Conditioning changed from ${snapshot.task} to ${current.task}.`);
  }
  return reasons;
}

/** A host-only change releases quick work from its old route without making the prompt stale. */
export function quickExpansionRouteIsCurrent(
  snapshot: QuickExpansionSnapshot,
  current: CurrentQuickExpansionInputs,
): boolean {
  if (current.selectedHostPolicy !== snapshot.selectedHostPolicy) return false;
  if (!current.readyHostIds.has(snapshot.route.hostId)) return false;
  const target = current.hostTargets?.get(snapshot.route.hostId);
  if (!target) return true;
  return (
    target.baseUrl === snapshot.route.target.baseUrl &&
    target.apiKey === snapshot.route.target.apiKey &&
    target.kind === snapshot.route.kind &&
    !knownInstanceIdsDiffer(snapshot.route.instanceId, target.instanceId)
  );
}

/** Monotonic guard used to reject late refreshes and discarded requests. */
export class PreparationRequestGuard {
  private token = 0;

  begin(): number {
    this.token += 1;
    return this.token;
  }

  invalidate(): void {
    this.token += 1;
  }

  isCurrent(token: number): boolean {
    return token === this.token;
  }
}
