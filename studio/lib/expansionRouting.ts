/*
 * Where prompt expansion runs.
 *
 * Expansion follows the generation route, but the generation router is
 * model-aware about the *checkpoint* — it knows nothing about the expansion
 * LLM. Under Auto / Most capable a batch can land on a machine that has the
 * checkpoint and not the expander, which 422s and offers a pull on exactly
 * that machine even when a peer already has it.
 *
 * This module is the one policy: prefer the generation route unless that host
 * is KNOWN to lack the expand model, otherwise re-rank the eligible machines
 * that positively have it. Eligibility mirrors the generation policy — every
 * ready machine under Auto / Most capable, only the pinned machine when the
 * user pinned one — and the ranking itself is handed in so this never becomes
 * a second copy of `pickAutoHost` / `pickMostCapableHost`.
 *
 * Absence of evidence is never absence of the model: an unread capability
 * snapshot (`model_present` undefined or null) keeps the generation route and
 * never qualifies a host as a reroute target.
 */

/** The manifest model local expansion uses when a host does not name one. */
export const DEFAULT_EXPAND_MODEL = "qwen3-expand";

/** Which machines expansion may consider, mirroring the generation policy. */
export type ExpansionPolicy =
  { kind: "auto" } | { kind: "capable" } | { kind: "pinned"; hostId: string };

/** The routing inputs for one machine. */
export interface ExpansionCandidate {
  hostId: string;
  /** False while the machine cannot accept a request at all. */
  ready: boolean;
  /**
   * `/api/capabilities.expand.model_present`. `undefined`/`null` means the
   * snapshot was never read or the host runs an API backend — unknown, never
   * "missing".
   */
  modelPresent?: boolean | null;
  /** `/api/capabilities.expand.configured`; `false` disqualifies the host. */
  configured?: boolean | null;
}

/**
 * Pick one host id out of an already-eligible set using the SAME ordering the
 * generation router uses for the active policy. Surfaces pass their own
 * `pickAutoHost` / `pickMostCapableHost` binding.
 */
export type ExpansionHostRanker = (hostIds: readonly string[]) => string | null;

export type ExpansionRouteDecision =
  /** Keep the generation route — it has the model, or nothing says it does not. */
  | { kind: "generation" }
  /** A different eligible machine positively has the expand model. */
  | { kind: "reroute"; hostId: string }
  /** No eligible machine has it: the caller offers the pull. */
  | { kind: "missing" };

function candidateFor(
  candidates: readonly ExpansionCandidate[],
  hostId: string | null | undefined,
): ExpansionCandidate | null {
  if (!hostId) return null;
  return candidates.find((candidate) => candidate.hostId === hostId) ?? null;
}

function usable(candidate: ExpansionCandidate): boolean {
  return candidate.ready && candidate.configured !== false;
}

/**
 * Resolve the host prompt expansion should run on.
 *
 * `generationRoute` is where the print itself would go; `null` when nothing is
 * routable. The decision is deliberately not a route object — the caller owns
 * building one from its own host store so per-host keys and instance identity
 * stay in the surface that holds them.
 */
export function resolveExpansionRoute(
  policy: ExpansionPolicy,
  generationRoute: { hostId: string } | null,
  candidates: readonly ExpansionCandidate[],
  rank: ExpansionHostRanker,
): ExpansionRouteDecision {
  const generation = candidateFor(candidates, generationRoute?.hostId);
  // Unknown is not missing: a host we have never read keeps the route it
  // already earned, exactly as before this policy existed.
  if (generationRoute && (!generation || generation.modelPresent !== false)) {
    return { kind: "generation" };
  }

  const eligible = candidates.filter((candidate) => {
    if (!usable(candidate)) return false;
    if (policy.kind === "pinned" && candidate.hostId !== policy.hostId)
      return false;
    return candidate.modelPresent === true;
  });
  if (eligible.length === 0) return { kind: "missing" };

  const chosen = rank(eligible.map((candidate) => candidate.hostId));
  if (!chosen) return { kind: "missing" };
  if (generationRoute && chosen === generationRoute.hostId)
    return { kind: "generation" };
  return { kind: "reroute", hostId: chosen };
}

/**
 * Normalize a surface's persisted generation-target pref into the policy.
 * Desktop stores `null` for Auto; web stores the `"auto"` sentinel.
 */
export function expansionPolicyForSelection(
  selection: string | null | undefined,
  sentinels: { auto?: string; capable?: string } = {},
): ExpansionPolicy {
  const auto = sentinels.auto ?? null;
  const capable = sentinels.capable ?? "capable";
  if (selection == null || selection === auto) return { kind: "auto" };
  if (selection === capable) return { kind: "capable" };
  return { kind: "pinned", hostId: selection };
}

/**
 * The engine's definitive missing-expansion-model error embeds its own fix:
 * "local expand model not found — run: mold pull qwen3-expand". Require both
 * halves before offering a pull: other expansion failures may mention that
 * command as recovery advice, but they do not prove the model is absent.
 */
export function parseMissingExpandModel(message: string): string | null {
  const match =
    /local expand model(?:\s+'[^']+')?\s+not found[^\n]*?mold pull ([\w.:@/-]+)/i.exec(
      message,
    );
  return match?.[1] ?? null;
}

/**
 * The expand model a host names, falling back to the manifest default for
 * servers whose `/api/capabilities.expand` predates the additive field.
 */
export function expandModelId(
  capability: { model?: string | null } | null | undefined,
): string {
  const named = capability?.model?.trim();
  return named ? named : DEFAULT_EXPAND_MODEL;
}
