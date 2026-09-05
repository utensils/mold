/**
 * Rough | Normal | Fine over the recipe's advertised octree allowlist. The
 * rungs are the server's; only the plain words are ours, and the mono octree
 * value stays beside them so the technical truth is never hidden.
 *
 * A ladder of exactly three rungs maps one-to-one. Any longer ladder shows its
 * floor, the advertised default and its ceiling — the three choices that mean
 * something to someone who is not counting voxels.
 */
export interface MeshDetailStep {
  value: number;
  label: string;
}

const LABELS_BY_COUNT: Record<number, readonly string[]> = {
  1: ["Normal"],
  2: ["Rough", "Fine"],
  3: ["Rough", "Normal", "Fine"],
};

export function meshDetailLadder(
  resolutions: readonly number[] | null | undefined,
  advertisedDefault: number | null | undefined,
): MeshDetailStep[] {
  const rungs = [...new Set(resolutions ?? [])].sort((a, b) => a - b);
  if (rungs.length === 0) return [];
  const picks =
    rungs.length === 3
      ? rungs
      : [
          ...new Set(
            [rungs[0]!, advertisedDefault ?? rungs[0]!, rungs[rungs.length - 1]!].filter((value) =>
              rungs.includes(value),
            ),
          ),
        ].sort((a, b) => a - b);
  const labels = LABELS_BY_COUNT[picks.length] ?? picks.map((value) => String(value));
  return picks.map((value, index) => ({ value, label: labels[index]! }));
}
