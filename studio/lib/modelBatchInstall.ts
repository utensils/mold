import type {
  ModelInstallAction,
  ModelInstallTarget,
} from "./modelInstallTargets";

export interface BatchInstallSelection<H> {
  modelId: string;
  targets: readonly ModelInstallTarget<H>[];
}

export interface BatchInstallItem {
  modelId: string;
  action: ModelInstallAction;
}

export interface BatchInstallHostPlan<H> {
  host: H;
  items: BatchInstallItem[];
  installCount: number;
  repairCount: number;
}

/** Both browser and desktop API clients retain an HTTP status, but use
 * different error classes. Treat a duplicate enqueue as idempotent without
 * coupling this shared planner to either surface's transport layer. */
export function isAlreadyQueuedError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "status" in error &&
    (error as { status?: unknown }).status === 409
  );
}

/**
 * Finds machines that can accept every selected model. The first selection's
 * target order stays authoritative so each surface retains its normal
 * install-first, repair-last machine ordering.
 */
export function planBatchInstallTargets<H extends { id: string }>(
  selections: readonly BatchInstallSelection<H>[],
): BatchInstallHostPlan<H>[] {
  const first = selections[0];
  if (!first) return [];

  return first.targets.flatMap((candidate) => {
    const items: BatchInstallItem[] = [];
    for (const selection of selections) {
      const target = selection.targets.find(
        ({ host }) => host.id === candidate.host.id,
      );
      if (!target) return [];
      items.push({ modelId: selection.modelId, action: target.action });
    }
    return [
      {
        host: candidate.host,
        items,
        installCount: items.filter(({ action }) => action === "install").length,
        repairCount: items.filter(({ action }) => action === "repair").length,
      },
    ];
  });
}
