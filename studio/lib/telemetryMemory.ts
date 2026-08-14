/**
 * Unified-memory detection for host telemetry panels.
 *
 * On Apple Silicon the GPU has no dedicated VRAM — Metal draws from the same
 * physical pool as system RAM, so a host-detail panel that renders both a
 * VRAM row and a RAM row shows the identical numbers twice. Every surface
 * derives "collapse those into one Memory row" from this one predicate
 * rather than re-inferring it per view.
 */
export function unifiedMemoryHost(
  gpus: ReadonlyArray<{ backend?: string | null; name?: string | null }>,
): boolean {
  if (gpus.length === 0) return false;
  return gpus.every((gpu) => {
    const backend = gpu.backend?.toLowerCase();
    // Older servers omit the additive backend field; an Apple GPU name is
    // the same inference the desktop host list already uses.
    if (!backend) return /\bapple\b/i.test(gpu.name ?? "");
    return backend === "metal";
  });
}
