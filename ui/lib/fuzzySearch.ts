/** Bounded, deterministic text matching for model selectors. */
function fold(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

/** Optimal-string-alignment distance: insertion/deletion/substitution/transposition. */
function distance(a: string, b: string): number {
  const x = [...a],
    y = [...b];
  let before = Array.from({ length: y.length + 1 }, (_, i) => i);
  let previous = before;
  for (let i = 1; i <= x.length; i++) {
    const row = [i];
    for (let j = 1; j <= y.length; j++) {
      row[j] = Math.min(
        row[j - 1]! + 1,
        previous[j]! + 1,
        previous[j - 1]! + Number(x[i - 1] !== y[j - 1]),
      );
      if (i > 1 && j > 1 && x[i - 1] === y[j - 2] && x[i - 2] === y[j - 1]) {
        row[j] = Math.min(row[j]!, before[j - 2]! + 1);
      }
    }
    before = previous;
    previous = row;
  }
  return previous[y.length]!;
}

function tokenScore(token: string, text: string): number {
  const words = text.split(" ");
  if (words.includes(token)) return 100;
  if (words.some((word) => word.startsWith(token))) return 80;
  if (text.includes(token)) return 60;
  if (text.replaceAll(" ", "").includes(token)) return 55;
  // Precision/version tokens must never turn Q4 into Q8 or 2.1 into 2.2.
  if (/\p{N}/u.test(token)) return 0;
  const letters = [...token];
  for (const word of words) {
    const chars = [...word];
    const limit = letters.length >= 8 ? 2 : letters.length >= 4 ? 1 : 0;
    if (
      limit &&
      chars.length <= 64 &&
      Math.abs(chars.length - letters.length) <= limit &&
      distance(token, word) <= limit
    )
      return 40;
  }
  // Ordered abbreviation inside ONE word, with bounded skipped letters.
  if (letters.length >= 2) {
    for (const word of words) {
      const chars = [...word];
      if (
        chars[0] !== letters[0] ||
        chars.length > letters.length * 2 ||
        chars.length - letters.length > 3
      )
        continue;
      let next = 0;
      for (const c of chars) if (c === letters[next]) next++;
      if (next === letters.length) return 20;
    }
  }
  return 0;
}

/** All tokens must match; metadata can help, but names win. Zero means no match. */
export function fuzzyTextScore(
  query: string,
  fields: readonly string[],
  metadata: readonly string[] = [],
): number {
  if (query.length > 128) return 0;
  const q = fold(query);
  if (!q) return 1;
  const tokens = q.split(" ");
  if (tokens.length > 12) return 0;
  const primary = fields.map((field) => fold(field.slice(0, 1024)));
  const secondary = metadata.map((field) => fold(field.slice(0, 1024)));
  let score = 0;
  for (const token of tokens) {
    const best = Math.max(
      0,
      ...primary.map((text) => tokenScore(token, text)),
      ...secondary.map((text) => Math.max(0, tokenScore(token, text) - 15)),
    );
    if (!best) return 0;
    score += best;
  }
  return score + (primary.includes(q) ? 1000 : 0);
}
