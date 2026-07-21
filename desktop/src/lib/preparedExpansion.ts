import type { HostRoute } from "../stores/hosts";

export type HostSelectionPolicy = string | null;

export interface PreparedExpansionInputs {
  sourcePrompt: string;
  model: string;
  family: string;
  requestedCount: number;
  selectedHostPolicy: HostSelectionPolicy;
}

export interface PreparedExpansionPrompt {
  id: string;
  text: string;
}

export interface PreparedExpansionBatch extends PreparedExpansionInputs {
  batchId: string;
  route: HostRoute;
  prompts: PreparedExpansionPrompt[];
}

export interface QuickExpansionSnapshot {
  requestToken: number;
  originalPrompt: string;
  expandedPrompt: string;
  model: string;
  family: string;
  selectedHostPolicy: HostSelectionPolicy;
  route: HostRoute;
}

export interface CurrentQuickExpansionInputs {
  expandedPrompt: string;
  model: string;
  family: string;
  selectedHostPolicy: HostSelectionPolicy;
  readyHostIds: ReadonlySet<string>;
  hostLabels: ReadonlyMap<string, string>;
  hostTargets?: CurrentPreparedExpansionInputs["hostTargets"];
}

export interface CurrentPreparedExpansionInputs extends PreparedExpansionInputs {
  readyHostIds: ReadonlySet<string>;
  hostLabels: ReadonlyMap<string, string>;
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
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `prepared-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function hostSelectionLabel(
  policy: HostSelectionPolicy,
  hostLabels: ReadonlyMap<string, string> = new Map(),
): string {
  if (policy === null) return "Auto";
  if (policy === "capable") return "Most capable";
  return hostLabels.get(policy) ?? policy;
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
    reasons.push(`Model changed from "${batch.model}" to "${current.model}".`);
  }
  if (current.family !== batch.family) {
    reasons.push(`Model family changed from "${batch.family}" to "${current.family}".`);
  }
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
    reasons.push(`Model changed from "${snapshot.model}" to "${current.model}".`);
  }
  if (current.family !== snapshot.family) {
    reasons.push(`Model family changed from "${snapshot.family}" to "${current.family}".`);
  }
  if (current.selectedHostPolicy !== snapshot.selectedHostPolicy) {
    reasons.push(
      `Host selection changed from ${hostSelectionLabel(snapshot.selectedHostPolicy, current.hostLabels)} to ${hostSelectionLabel(current.selectedHostPolicy, current.hostLabels)}.`,
    );
  }
  if (!current.readyHostIds.has(snapshot.route.hostId)) {
    reasons.push(`${snapshot.route.label} is no longer reachable.`);
  } else {
    const target = current.hostTargets?.get(snapshot.route.hostId);
    if (
      target &&
      (target.baseUrl !== snapshot.route.target.baseUrl ||
        target.apiKey !== snapshot.route.target.apiKey ||
        target.kind !== snapshot.route.kind ||
        knownInstanceIdsDiffer(snapshot.route.instanceId, target.instanceId))
    ) {
      reasons.push(`${snapshot.route.label}'s connection details changed.`);
    }
  }
  return reasons;
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
