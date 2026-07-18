/**
 * Pure selection-model helpers for the gallery's bulk select mode.
 *
 * Selection is keyed by print identity (filename — the cross-host identity
 * in the merged grid), never by row index: the virtualized justified grid
 * re-flows rows on resize and refetch, so indexes are not stable.
 */

export interface SelectionClickOptions {
  /** Shift was held: extend the inclusive range from the anchor. */
  shift?: boolean;
  /** Cmd/Ctrl was held: toggle without collapsing (same as a plain click
   *  in select mode — accepted so muscle memory from other apps works). */
  meta?: boolean;
}

export interface SelectionState {
  selection: Set<string>;
  anchor: string | null;
}

/**
 * Apply one click in select mode, Finder-style:
 * - plain / meta click toggles the item and re-anchors on it;
 * - shift-click selects the inclusive range between the anchor and the
 *   clicked item in the current filter order, adding to (never clearing)
 *   the existing selection, and keeps the anchor;
 * - shift-click with a stale anchor (filter changed underneath) falls back
 *   to a single toggle.
 *
 * Never mutates the incoming set.
 */
export function applySelectionClick(
  current: ReadonlySet<string>,
  anchor: string | null,
  order: readonly string[],
  clicked: string,
  opts: SelectionClickOptions = {},
): SelectionState {
  if (opts.shift && anchor !== null) {
    const a = order.indexOf(anchor);
    const b = order.indexOf(clicked);
    if (a !== -1 && b !== -1) {
      const [lo, hi] = a < b ? [a, b] : [b, a];
      const selection = new Set(current);
      for (let i = lo; i <= hi; i++) selection.add(order[i]!);
      return { selection, anchor };
    }
  }
  const selection = new Set(current);
  if (selection.has(clicked)) selection.delete(clicked);
  else selection.add(clicked);
  return { selection, anchor: clicked };
}
