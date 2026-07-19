import type { ModelEntry } from "./api/types";

export interface HostedModelEntry extends ModelEntry {
  hostIds: string[];
}

export function mergeInstalledModels(
  primary: ModelEntry[],
  hosted: HostedModelEntry[],
): ModelEntry[] {
  const byName = new Map<string, ModelEntry>();
  for (const entry of primary) byName.set(entry.name, entry);
  for (const entry of hosted) if (!byName.has(entry.name)) byName.set(entry.name, entry);
  return [...byName.values()];
}

/**
 * Narrow the model picker to the sticky target host's installed set. Auto
 * (`null`) and the "capable" sentinel route per-job, so they keep the all-host union;
 * `availableNames === null` means that host's list was never fetched — show
 * the union rather than an empty picker built from missing data.
 */
export function filterModelsForTarget(
  installed: ModelEntry[],
  target: string | null,
  availableNames: ReadonlySet<string> | null,
): ModelEntry[] {
  if (!target || target === "capable" || availableNames === null) return installed;
  return installed.filter((entry) => availableNames.has(entry.name));
}

export function findInstalledModel(installed: ModelEntry[], name: string): ModelEntry | null {
  return installed.find((entry) => entry.name === name) ?? null;
}

export function preferredInstalledModel(installed: ModelEntry[]): ModelEntry | null {
  return installed.find((entry) => entry.family === "flux") ?? installed[0] ?? null;
}

export function shouldShowStarterCards(input: {
  connectionReady: boolean;
  primaryLoading: boolean;
  hostsInitialized: boolean;
  hostModelsLoading: boolean;
  allReadyHostsFetched: boolean;
  installed: ModelEntry[];
}): boolean {
  return (
    input.connectionReady &&
    !input.primaryLoading &&
    input.hostsInitialized &&
    !input.hostModelsLoading &&
    input.allReadyHostsFetched &&
    input.installed.length === 0
  );
}
