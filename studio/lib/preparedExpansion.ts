/*
 * Why reviewed prompt work no longer matches the form — ONE rule for every
 * surface.
 *
 * Desktop froze a prepared batch or a quick rewrite against the inputs that
 * produced it and named each divergence in a stable sentence; web grew its own
 * copy of the same list in `CreatePage.vue`, which drifted (it said "the Run on
 * selection changed" where desktop said the selection had changed from one
 * policy to another). A reason a person reads on two screens must be the same
 * sentence, so the rule lives here and both surfaces bind their own state to
 * it.
 *
 * The route types are STRUCTURAL: `studio/` may not import a shell's host
 * store, and web's `HostRoute` and desktop's are two different declarations of
 * the same facts.
 */
import type { ExpandTask } from "./expandTask";
import type { RemixDimension } from "./promptTransform";

/** `null` = Auto, `"capable"` = Most capable, anything else pins a machine. */
export type HostSelectionPolicy = string | null;

/** The frozen machine a reviewed rewrite was prepared against. */
export interface StaleRouteSnapshot {
  hostId: string;
  label: string;
  kind?: string | null;
  instanceId?: string | null;
  target: { baseUrl: string; apiKey?: string | null };
}

/** What the registry says about that machine right now. */
export interface StaleHostTarget {
  baseUrl: string;
  apiKey?: string | null;
  kind?: string | null;
  instanceId?: string | null;
}

/** The facts a reviewed rewrite freezes, and the live ones it is compared to. */
export interface PreparedExpansionStaleFacts {
  kind?: "expand" | "remix";
  sourcePrompt: string;
  dimensions?: readonly RemixDimension[];
  conditioningFingerprint?: string;
  model: string;
  family: string;
  task: ExpandTask;
  requestedCount: number;
  selectedHostPolicy: HostSelectionPolicy;
}

export interface PreparedExpansionStaleBatch extends PreparedExpansionStaleFacts {
  route: StaleRouteSnapshot;
}

export interface PreparedExpansionStaleInputs extends PreparedExpansionStaleFacts {
  readyHostIds: ReadonlySet<string>;
  hostLabels: ReadonlyMap<string, string>;
  /** Plain style names, so a reason reads "Photoreal" rather than an id. */
  modelLabels?: ReadonlyMap<string, string>;
  hostTargets?: ReadonlyMap<string, StaleHostTarget>;
}

export interface QuickExpansionStaleSnapshot {
  expandedPrompt: string;
  model: string;
  family: string;
  task: ExpandTask;
}

export interface QuickExpansionStaleInputs extends QuickExpansionStaleSnapshot {
  modelLabels?: ReadonlyMap<string, string>;
}

function modelLabel(
  name: string,
  labels?: ReadonlyMap<string, string>,
): string {
  return labels?.get(name) ?? name;
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
  batch: PreparedExpansionStaleBatch,
  current: PreparedExpansionStaleInputs,
): string[] {
  const reasons: string[] = [];
  if (current.sourcePrompt !== batch.sourcePrompt) {
    reasons.push("Source prompt changed after these variations were prepared.");
  }
  if (current.model !== batch.model) {
    reasons.push(
      `Style changed from "${modelLabel(batch.model, current.modelLabels)}" to "${modelLabel(current.model, current.modelLabels)}".`,
    );
  }
  if (current.family !== batch.family) {
    reasons.push(
      `Style family changed from "${batch.family}" to "${current.family}".`,
    );
  }
  if (current.task !== batch.task) {
    reasons.push(`Conditioning changed from ${batch.task} to ${current.task}.`);
  }
  if (
    batch.conditioningFingerprint !== undefined &&
    current.conditioningFingerprint !== batch.conditioningFingerprint
  ) {
    reasons.push(
      "Conditioning media changed after these variations were prepared.",
    );
  }
  if (
    batch.kind === "remix" &&
    JSON.stringify(current.dimensions ?? []) !==
      JSON.stringify(batch.dimensions ?? [])
  ) {
    reasons.push(
      "Remix dimensions changed after these variations were prepared.",
    );
  }
  if (current.requestedCount !== batch.requestedCount) {
    reasons.push(
      `Batch changed from ${batch.requestedCount} to ${current.requestedCount}.`,
    );
  }
  if (current.selectedHostPolicy !== batch.selectedHostPolicy) {
    reasons.push(
      `Machine selection changed from ${hostSelectionLabel(batch.selectedHostPolicy, current.hostLabels)} to ${hostSelectionLabel(current.selectedHostPolicy, current.hostLabels)}.`,
    );
  }
  if (!current.readyHostIds.has(batch.route.hostId)) {
    reasons.push(`${batch.route.label} is no longer reachable.`);
  } else {
    const currentTarget = current.hostTargets?.get(batch.route.hostId);
    if (
      currentTarget &&
      ((currentTarget.baseUrl ?? null) !==
        (batch.route.target.baseUrl ?? null) ||
        (currentTarget.apiKey ?? null) !==
          (batch.route.target.apiKey ?? null) ||
        (currentTarget.kind ?? null) !== (batch.route.kind ?? null) ||
        knownInstanceIdsDiffer(
          batch.route.instanceId,
          currentTarget.instanceId,
        ))
    ) {
      reasons.push(`${batch.route.label}'s connection details changed.`);
    }
  }
  return reasons;
}

/** Batch 1 gets the same frozen-route guarantees without a review workspace. */
export function quickExpansionStaleReasons(
  snapshot: QuickExpansionStaleSnapshot,
  current: QuickExpansionStaleInputs,
): string[] {
  const reasons: string[] = [];
  if (current.expandedPrompt !== snapshot.expandedPrompt) {
    reasons.push("Expanded prompt changed after it was prepared.");
  }
  if (current.model !== snapshot.model) {
    reasons.push(
      `Style changed from "${modelLabel(snapshot.model, current.modelLabels)}" to "${modelLabel(current.model, current.modelLabels)}".`,
    );
  }
  if (current.family !== snapshot.family) {
    reasons.push(
      `Style family changed from "${snapshot.family}" to "${current.family}".`,
    );
  }
  if (current.task !== snapshot.task) {
    reasons.push(
      `Conditioning changed from ${snapshot.task} to ${current.task}.`,
    );
  }
  return reasons;
}
