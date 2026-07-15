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
