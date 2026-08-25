import type { PromptTransformProvenance, RemixDimension } from "../lib/api/types";
import type { PreparedExpansionBatch, QuickExpansionSnapshot } from "../lib/preparedExpansion";
import type { HostRoute } from "../stores/hosts";

export interface MobilePreparedSubmissionSnapshot {
  batchId: string;
  promptIds: string[];
  prompts: string[];
  originalPrompt: string;
  promptTransforms?: PromptTransformProvenance[];
  route: HostRoute;
}

export interface MobileQuickSubmissionSnapshot {
  requestToken: number;
  route: HostRoute;
}

function cloneRoute(route: HostRoute): HostRoute {
  return { ...route, target: { ...route.target } };
}

function remixDimensions(batch: PreparedExpansionBatch, index: number): RemixDimension[] {
  const perVariant = (
    batch as PreparedExpansionBatch & {
      remixVariantDimensions?: readonly (readonly RemixDimension[])[];
    }
  ).remixVariantDimensions;
  return [...(perVariant?.[index] ?? batch.dimensions ?? [])];
}

export function capturePreparedSubmission(
  prepared: PreparedExpansionBatch | null,
): MobilePreparedSubmissionSnapshot | null {
  if (!prepared) return null;
  const promptTransforms =
    prepared.kind === "remix"
      ? prepared.prompts.map((_, index): PromptTransformProvenance => ({
          operation: "remix",
          ...(prepared.rootPrompt ? { root_prompt: prepared.rootPrompt } : {}),
          source_prompt: prepared.sourcePrompt,
          source_kind: prepared.sourceKind ?? "current",
          task: prepared.task,
          dimensions: remixDimensions(prepared, index),
        }))
      : null;
  return {
    batchId: prepared.batchId,
    promptIds: prepared.prompts.map((prompt) => prompt.id),
    prompts: prepared.prompts.map((prompt) => prompt.text.trim()),
    originalPrompt: prepared.rootPrompt ?? prepared.sourcePrompt,
    ...(promptTransforms ? { promptTransforms } : {}),
    route: cloneRoute(prepared.route),
  };
}

export function captureQuickSubmission(
  quick: QuickExpansionSnapshot | null,
): MobileQuickSubmissionSnapshot | null {
  return quick ? { requestToken: quick.requestToken, route: cloneRoute(quick.route) } : null;
}

export function preparedSubmissionIsCurrent(
  snapshot: MobilePreparedSubmissionSnapshot,
  current: PreparedExpansionBatch | null,
  staleReasons: readonly string[],
): boolean {
  return (
    staleReasons.length === 0 &&
    current?.batchId === snapshot.batchId &&
    current.prompts.length === snapshot.prompts.length &&
    current.prompts.every(
      (prompt, index) =>
        prompt.id === snapshot.promptIds[index] && prompt.text.trim() === snapshot.prompts[index],
    )
  );
}

export function quickSubmissionIsCurrent(
  snapshot: MobileQuickSubmissionSnapshot,
  current: QuickExpansionSnapshot | null,
  staleReasons: readonly string[],
): boolean {
  return staleReasons.length === 0 && current?.requestToken === snapshot.requestToken;
}
