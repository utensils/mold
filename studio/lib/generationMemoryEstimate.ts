/**
 * Cross-surface policy for the advisory Create VRAM badge.
 *
 * Admission still uses live free memory. The badge answers the more durable
 * question: whether this request fits the selected host's hardware once work
 * ahead of it releases its allocations. Never derive this verdict from the
 * moment-to-moment free-memory fields.
 */

export interface GenerationCapacityEstimate {
  peak_memory_bytes: number;
  available_memory_bytes?: number | null;
  fits_available_memory?: boolean | null;
  capacity_peak_memory_bytes?: number | null;
  device_capacity_bytes?: number | null;
  fits_device_capacity?: boolean | null;
}

export type EstimateFit = "fits" | "tight" | "wont-fit" | "unknown";

export function classifyGenerationFit(
  est: GenerationCapacityEstimate,
): EstimateFit {
  const peak = est.capacity_peak_memory_bytes;
  const capacity = est.device_capacity_bytes;
  if (peak == null || capacity == null || capacity <= 0) return "unknown";
  if (est.fits_device_capacity === false) return "wont-fit";
  if (est.fits_device_capacity == null) return "unknown";
  if (peak > capacity * 0.8) return "tight";
  return "fits";
}

export function displayGenerationMemory(est: GenerationCapacityEstimate): {
  peakBytes: number;
  capacityBytes: number | null;
} {
  return {
    peakBytes: est.capacity_peak_memory_bytes ?? est.peak_memory_bytes,
    capacityBytes:
      est.capacity_peak_memory_bytes != null
        ? (est.device_capacity_bytes ?? null)
        : null,
  };
}

const gb = (bytes: number) => `${(bytes / 1_000_000_000).toFixed(1)} GB`;

export function generationEstimateLabel(
  fit: EstimateFit,
  peakBytes: number,
  capacityBytes: number | null,
): string {
  switch (fit) {
    case "fits":
      return capacityBytes !== null
        ? `VRAM · fits — est. ${gb(peakBytes)} of ${gb(capacityBytes)}`
        : `VRAM · est. ${gb(peakBytes)}`;
    case "tight":
      return "VRAM · tight — close other apps";
    case "wont-fit":
      return "VRAM · won't fit on this GPU";
    default:
      return `VRAM · est. ${gb(peakBytes)}`;
  }
}

export const GENERATION_ESTIMATE_TOOLTIP =
  "Estimated peak GPU memory (VRAM) this print needs versus the connected " +
  "GPU's physical capacity. Active and queued work do not change this hardware-fit verdict.";
