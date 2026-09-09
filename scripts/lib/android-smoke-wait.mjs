// The waiting policy both Android smoke scripts share.
//
// It lives in one module because each rule below was got wrong by a copy that
// only existed in one of the two scripts.

/** How long a single wait may take before it is called a failure. */
export const DEFAULT_SMOKE_TIMEOUT_MS = 30_000;
/** How often a wait re-reads its condition. */
export const POLL_INTERVAL_MS = 200;
/** How many polls pass before a retried action is dispatched again (~1s). */
export const REDISPATCH_EVERY_ATTEMPTS = 5;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Read the deadline out of the environment.
 *
 * `Number(process.env.X ?? fallback)` looks right and is not: `??` only catches
 * `null`/`undefined`, so an EMPTY variable parses as `0` — and a zero deadline
 * makes `while (Date.now() < deadline)` false on its first evaluation, so every
 * wait fails after zero attempts with no clue why. `MOLD_ANDROID_SMOKE_TIMEOUT_MS=`
 * in a shell, or an unset `env:` value in a workflow, is exactly that string.
 *
 * Absent or blank falls back. A value that is PRESENT but unusable throws
 * instead, because silently ignoring a typo would run the suite on a deadline
 * nobody chose.
 */
export function resolveDeadlineMs(raw, fallback = DEFAULT_SMOKE_TIMEOUT_MS) {
  if (raw === undefined || raw === null || String(raw).trim() === "")
    return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0)
    throw new Error(
      `MOLD_ANDROID_SMOKE_TIMEOUT_MS must be a positive number of ` +
        `milliseconds, got ${JSON.stringify(String(raw))}`,
    );
  return parsed;
}

/**
 * Poll `read` until it answers truthy, then return that value.
 *
 * The failure says what was waited on, for how long, and how many times it
 * looked: "Timed out: select hosts" alone could not tell a slow emulator from
 * a condition that was never going to hold.
 */
export async function until(read, label, options = {}) {
  const timeoutMs = options.timeoutMs ?? DEFAULT_SMOKE_TIMEOUT_MS;
  const now = options.now ?? (() => Date.now());
  const pause = options.sleep ?? sleep;
  const started = now();
  const deadline = started + timeoutMs;
  // Kept apart on purpose: a re-dispatch that throws must not clobber the
  // reason the wait was actually failing. Re-tapping a button that has since
  // been `v-if`'d away throws every single time, and folding both into one
  // slot left every genuine timeout blaming the retry.
  let lastReadError;
  let lastActError;
  let attempts = 0;
  while (now() < deadline) {
    attempts += 1;
    try {
      const value = await read();
      if (value) return value;
    } catch (error) {
      lastReadError = error;
    }
    if (options.onRetry) {
      try {
        await options.onRetry(attempts);
      } catch (error) {
        lastActError = error;
      }
    }
    await pause(options.pollIntervalMs ?? POLL_INTERVAL_MS);
  }
  const failure = new Error(
    `Timed out: ${label} (${Math.round((now() - started) / 1000)}s, ` +
      `${attempts} attempts)`,
    { cause: lastReadError },
  );
  if (lastActError) failure.actError = lastActError;
  throw failure;
}

/**
 * Do something, then wait for it to have taken effect — re-doing it if it did
 * not.
 *
 * A synthetic `Input.dispatchTouchEvent` is occasionally dropped on a CI
 * emulator: the evidence from every timed-out run shows the app still on the
 * PREVIOUS tab with focus still on the previous tab's button, rendered,
 * responsive, and with no ANR, while the wait round-tripped ~150 CDP calls
 * against a screen no tap had reached. Retrying only the ASSERTION cannot
 * recover from that — the tap is never sent again — which is why a longer
 * deadline bought nothing but a slower red build. The action itself is what
 * has to be retried.
 */
export async function actUntil(act, read, label, options = {}) {
  const every = options.redispatchEvery ?? REDISPATCH_EVERY_ATTEMPTS;
  await act();
  return until(read, label, {
    ...options,
    onRetry: async (attempts) => {
      if (attempts % every !== 0) return;
      // Re-READ, immediately before re-dispatching. The poll that scheduled
      // this retry is already one interval old, and for an action that is not
      // idempotent that gap is the whole risk: a second Android Back arriving
      // after the panel closed finds nothing to consume, so `useMobileBack`
      // returns without `preventDefault`, the native plugin sees
      // `consumed != "true"` and calls `delegate()` — which finishes the
      // activity. The smoke test would then PASS its Back assertion (the tab
      // bar is back) and fail three steps later for no visible reason.
      if (await read()) return;
      await act();
    },
  });
}
