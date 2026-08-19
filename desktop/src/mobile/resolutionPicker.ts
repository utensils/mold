/** Desktop generation also snaps manual dimensions to the engine's latent stride. */
export function snapMobileDimension(value: number, minimum = 64, alignment = 16): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return minimum;
  const grid = Math.max(1, Math.round(alignment));
  return Math.max(minimum, Math.round(numeric / grid) * grid);
}
