/**
 * Operation-count budgets for the gallery regression guards.
 *
 * Wall-clock timings are noisy in CI, so every Library performance guard
 * asserts a COUNT of expensive operations (union passes, identity scans,
 * component mounts, queue sorts) against a budget expressed in terms of the
 * data size. A guard that fails names the budget it broke and by how much,
 * so a regression reads as "unionOrganization ran 12 000 times for 2 000
 * prints (budget 2 000)" rather than "the test got slower".
 */

/** Throws a descriptive error when `observed` exceeds `budget`. */
export function expectOpsUnder(
  label: string,
  observed: number,
  budget: number,
): void {
  if (observed <= budget) return;
  const over = observed - budget;
  throw new Error(
    `${label}: ${observed} operations exceeds the budget of ${budget} (+${over}). ` +
      "A gallery hot path is doing per-item work that must be indexed once per data change.",
  );
}

/** Counts invocations of a function while delegating to it unchanged. */
export function countedFn<A extends unknown[], R>(
  fn: (...args: A) => R,
): { fn: (...args: A) => R; count: () => number; reset: () => void } {
  let calls = 0;
  return {
    fn: (...args: A) => {
      calls += 1;
      return fn(...args);
    },
    count: () => calls,
    reset: () => {
      calls = 0;
    },
  };
}
