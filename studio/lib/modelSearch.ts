/*
 * Model search for the ⌘K palette.
 *
 * The palette already listed installed models, but only the ones the serving
 * host happened to report, and a model you did not have was simply absent —
 * the user had to know it existed, leave the palette, open Models ▸ Discover,
 * search again, pick a machine, and come back. This module makes one query
 * cover all three cases a user actually means by "switch to that model":
 *
 *   - it is here            → use it
 *   - it is on another box  → pin generation there, then use it
 *   - nobody has it         → pull it onto a machine that can take it
 *
 * Everything here is pure so the policy is testable without a Vue runtime and
 * identical on web and desktop. Surfaces own the async catalog fetch, the
 * install dispatch, and their own ranking; this module owns what a row says,
 * what it means, and which machine it points at.
 */

/** Shortest query worth a live catalog round trip. */
export const CATALOG_SEARCH_MIN_QUERY = 2;
/** Catalog rows are appended after the ranked commands — keep them a tail, not a wall. */
export const CATALOG_SEARCH_LIMIT = 5;
/** Matches the Discover surfaces closely enough to feel like the same search. */
export const CATALOG_SEARCH_DEBOUNCE_MS = 250;

/** The slice of an installed `/api/models` row the palette reasons over. */
export interface SearchableModel {
  name: string;
  display_name?: string | null;
  family?: string | null;
  is_loaded?: boolean | null;
}

/** The slice of a live catalog entry the palette reasons over. */
export interface SearchableCatalogEntry {
  /** Stable `cv:` / `hf:` id — the value both download routes want. */
  id: string;
  name: string;
  family?: string | null;
  source?: string | null;
  /** The catalog's own "the serving host has this" flag, when it reports one. */
  installed?: boolean | null;
}

interface ModelCommandBase {
  id: string;
  label: string;
  /** Right-aligned meta: the machine, the load state, or the family. */
  hint: string;
  keywords: string[];
}

export interface UseModelCommand extends ModelCommandBase {
  kind: "use";
  /** Raw model id for the request and for persistence — never the display name. */
  model: string;
  /** Machine to pin generation to first, or null when routing already reaches it. */
  switchToHostId: string | null;
}

export interface InstallModelCommand extends ModelCommandBase {
  kind: "install";
  /** `cv:` / `hf:` id or manifest name, exactly as the download routes want it. */
  model: string;
  displayName: string;
}

export interface HostResolution {
  switchToHostId: string | null;
  /** The machine a job for this model would land on, or null if nobody has it. */
  ownerHostId: string | null;
}

/**
 * Where a "use this model" row points.
 *
 * Automatic routing (web's Auto / Most capable, desktop's null target) is
 * already model-aware, so it needs no switch even when the model lives on
 * another box. A *pinned* target that lacks the model is the one case worth
 * moving the user: silently generating somewhere else would contradict the
 * pin, and refusing would make the palette useless for the fleet.
 */
export function resolveModelHost(
  owners: readonly string[],
  targetHostId: string | null | undefined,
  options: { autoTargetIds?: readonly string[] } = {},
): HostResolution {
  const ownerHostId = owners[0] ?? null;
  if (ownerHostId === null) return { switchToHostId: null, ownerHostId: null };

  const auto = options.autoTargetIds ?? [];
  const pinned =
    targetHostId != null && targetHostId !== "" && !auto.includes(targetHostId);
  if (!pinned || owners.includes(targetHostId as string)) {
    return { switchToHostId: null, ownerHostId };
  }
  return { switchToHostId: ownerHostId, ownerHostId };
}

/** `flux-dev:q4` → `["flux-dev:q4", "flux", "dev", "q4"]`. */
function tokens(value: string): string[] {
  const trimmed = value.trim();
  if (!trimmed) return [];
  const parts = trimmed.split(/[\s\-_.:/]+/).filter(Boolean);
  return parts.length > 1 ? [trimmed, ...parts] : [trimmed];
}

function unique(values: readonly (string | null | undefined)[]): string[] {
  const seen = new Set<string>();
  for (const value of values) {
    const key = value?.trim();
    if (key) seen.add(key);
  }
  return [...seen];
}

function displayNameOf(model: SearchableModel): string {
  const display = model.display_name?.trim();
  return display ? display : model.name;
}

export interface UseModelCommandOptions {
  /** Machines known to hold the model on disk, in preference order. */
  ownersFor: (name: string) => readonly string[];
  /** The pinned generation target, or an automatic sentinel / null. */
  targetHostId?: string | null;
  /** Sentinels that mean "let the router decide" (web's `auto` / `capable`). */
  autoTargetIds?: readonly string[];
  hostLabel?: (hostId: string) => string;
}

/** Installed models become "Use <name>" rows, ranked by the surface's matcher. */
export function buildUseModelCommands(
  models: readonly SearchableModel[],
  options: UseModelCommandOptions,
): UseModelCommand[] {
  const hostLabel = options.hostLabel ?? ((id: string) => id);
  return models.map((model) => {
    const title = displayNameOf(model);
    const { switchToHostId } = resolveModelHost(
      options.ownersFor(model.name),
      options.targetHostId ?? null,
      { autoTargetIds: options.autoTargetIds ?? [] },
    );
    const hint = switchToHostId
      ? `on ${hostLabel(switchToHostId)}`
      : model.is_loaded
        ? "loaded"
        : (model.family?.trim() ?? "");
    return {
      kind: "use",
      id: `model-${model.name}`,
      model: model.name,
      label: `Use ${title}`,
      hint,
      switchToHostId,
      keywords: unique([
        ...tokens(model.name),
        ...tokens(title),
        ...tokens(model.family ?? ""),
        "model",
        "use",
        "switch",
      ]),
    };
  });
}

export interface InstallModelCommandOptions {
  /** Model ids AND server-reported names already installed on some machine. */
  installedNames?: Iterable<string>;
  limit?: number;
}

/** Catalog hits nobody has yet become "Install <name>" rows. */
export function buildInstallModelCommands(
  entries: readonly SearchableCatalogEntry[],
  options: InstallModelCommandOptions = {},
): InstallModelCommand[] {
  // A catalog row and an installed row can disagree on which identifier they
  // lead with (opaque id vs. the human title the sidecar recorded), so both are
  // treated as evidence of ownership rather than offering a duplicate pull.
  const installed = new Set(options.installedNames ?? []);
  const limit = options.limit ?? CATALOG_SEARCH_LIMIT;
  const rows: InstallModelCommand[] = [];
  for (const entry of entries) {
    if (rows.length >= limit) break;
    if (entry.installed) continue;
    if (installed.has(entry.id) || installed.has(entry.name)) continue;
    const source = entry.source?.trim();
    rows.push({
      kind: "install",
      id: `install-${entry.id}`,
      model: entry.id,
      displayName: entry.name,
      label: `Install ${entry.name}`,
      hint: source ? `not installed · ${source}` : "not installed",
      keywords: unique([
        ...tokens(entry.name),
        ...tokens(entry.family ?? ""),
        "install",
        "download",
        "pull",
        "model",
      ]),
    });
  }
  return rows;
}

/** True once the query is long enough that a catalog search is worth issuing. */
export function shouldSearchCatalog(query: string): boolean {
  return query.trim().length >= CATALOG_SEARCH_MIN_QUERY;
}
