/**
 * Command-palette matching. Pure and UI-agnostic: the ⌘K component builds a
 * registry of commands (each carrying a `run` closure over router/stores) and
 * this module ranks them against the query. Ranking is prefix > word-start >
 * substring, ties broken by original registry order (stable).
 */

export interface Matchable {
  title: string;
  /** Extra text to match against (e.g. model family, synonyms). */
  keywords?: string[];
}

/** 3 = prefix, 2 = word-start, 1 = substring, 0 = no match. */
export function scoreText(query: string, text: string): number {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  if (t.startsWith(q)) return 3;
  if (t.split(/[\s\-_.:/]+/).some((word) => word.startsWith(q))) return 2;
  if (t.includes(q)) return 1;
  return 0;
}

/**
 * Rank `items` against `query`. An empty query returns the items unchanged
 * (stable registry order). Non-matching items are dropped; ties keep their
 * original order.
 */
export function matchCommands<T extends Matchable>(query: string, items: T[]): T[] {
  const q = query.trim();
  if (!q) return items.slice();
  return items
    .map((item, index) => {
      const texts = [item.title, ...(item.keywords ?? [])];
      const score = Math.max(...texts.map((t) => scoreText(q, t)));
      return { item, score, index };
    })
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map((s) => s.item);
}
