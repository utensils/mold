/** Keep viewing the same filtered list when its current media leaves it.
 * Object callers supply their existing copy identity, including renamed mirrors.
 */
export function viewerAfterRemoval<T>(
  current: T | undefined,
  previous: readonly T[],
  remaining: readonly T[],
  same: (a: T, b: T) => boolean = Object.is,
): T | undefined {
  if (current === undefined) return undefined;
  const survivor = (item: T) =>
    remaining.find((candidate) => same(item, candidate));
  const retained = survivor(current);
  if (retained !== undefined) return retained;
  const index = previous.findIndex((item) => same(current, item));
  if (index >= 0) {
    for (let i = index + 1; i < previous.length; i++) {
      const next = survivor(previous[i]!);
      if (next !== undefined) return next;
    }
    for (let i = index - 1; i >= 0; i--) {
      const next = survivor(previous[i]!);
      if (next !== undefined) return next;
    }
  }
  return remaining[0];
}
