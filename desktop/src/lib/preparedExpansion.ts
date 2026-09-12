import type { HostRoute } from "../stores/hosts";
import { createUuid } from "@studio/lib/id";
import type { ExpandContext, ExpandTask } from "@studio/lib/expandTask";
import { type RemixDimension, type RemixSourceKind } from "@studio/lib/promptTransform";
import type { PromptTransformProvenance } from "./api/types";

/*
 * The stale-reason sentences are ONE rule, shared with web:
 * `@studio/lib/preparedExpansion`. What stays here is the desktop-only
 * machinery around them — the `HostRoute`-typed batch, its id, the route
 * currency check and the request guard.
 */
export { hostSelectionLabel, type HostSelectionPolicy } from "@studio/lib/preparedExpansion";
import type { HostSelectionPolicy } from "@studio/lib/preparedExpansion";
import {
  preparedExpansionStaleReasons as sharedPreparedExpansionStaleReasons,
  quickExpansionStaleReasons as sharedQuickExpansionStaleReasons,
} from "@studio/lib/preparedExpansion";

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
  /** Frozen generation facts sent with the rewrite (additive). */
  context?: ExpandContext;
  requestedCount: number;
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

// The count-and-normalise rule is the shared studio helper; web and desktop
// must agree on the messages and on unwrapping a one-string JSON array.
export { validateExpandedPrompts } from "@studio/lib/expandedPrompts";

/** Specific, stable reasons why reviewed work no longer matches the form. */
export function preparedExpansionStaleReasons(
  batch: PreparedExpansionBatch,
  current: CurrentPreparedExpansionInputs,
): string[] {
  return sharedPreparedExpansionStaleReasons(batch, current);
}

/** Batch 1 gets the same frozen-route guarantees without a review workspace. */
export function quickExpansionStaleReasons(
  snapshot: QuickExpansionSnapshot,
  current: CurrentQuickExpansionInputs,
): string[] {
  return sharedQuickExpansionStaleReasons(snapshot, current);
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

function knownInstanceIdsDiffer(
  frozen: string | null | undefined,
  current: string | null | undefined,
): boolean {
  return frozen != null && current != null && frozen !== current;
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
  private controller: AbortController | null = null;

  begin(): number {
    this.controller?.abort(new Error("superseded"));
    this.token += 1;
    this.controller = new AbortController();
    return this.token;
  }

  invalidate(): void {
    this.controller?.abort(new Error("cancelled"));
    this.controller = null;
    this.token += 1;
  }

  signalFor(token: number): AbortSignal {
    if (token !== this.token || !this.controller) {
      throw new Error("request token is no longer current");
    }
    return this.controller.signal;
  }

  isCurrent(token: number): boolean {
    return token === this.token;
  }
}
