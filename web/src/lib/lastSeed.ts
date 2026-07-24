/**
 * Persistence + eligibility rules for Create's "lock last (seed)" affordance.
 *
 * The value can't live on the job rail alone: completed cards auto-dismiss
 * ~1.5 s after landing and the persistence watcher then writes the shortened
 * list, so a reload outside that grace window would forget the seed. A tiny
 * dedicated key keeps the latest lockable seed durable across sessions.
 */

export const LAST_SEED_STORAGE_KEY = "mold.web.lastSeed.v1";

/** A lockable seed is a finite non-negative integer-ish number. */
function validSeed(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

export function loadLastSeed(): number | null {
  try {
    const raw = localStorage.getItem(LAST_SEED_STORAGE_KEY);
    if (raw === null) return null;
    const parsed = Number(raw);
    return validSeed(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function storeLastSeed(seed: number): void {
  try {
    localStorage.setItem(LAST_SEED_STORAGE_KEY, String(seed));
  } catch {
    /* quota / privacy mode — the in-memory ref still has it */
  }
}

/**
 * The seed a completed job offers for locking, or `null` when it has none.
 *
 * Chain completions fabricate `seed_used` (`chainCompleteToSingle` falls back
 * to `request.seed ?? 0`), so a chain that ran with a random seed reports 0 —
 * never a real seed to lock. A chain whose request pinned an explicit seed is
 * genuine; single-clip completions always carry the server-reported seed.
 */
export function completedSeedToLock(job: {
  chain: unknown | null;
  request: { seed?: number | null };
  result: { seed_used: number } | null;
}): number | null {
  const seed = job.result?.seed_used;
  if (!validSeed(seed)) return null;
  if (job.chain != null && job.request.seed == null) return null;
  return seed;
}
