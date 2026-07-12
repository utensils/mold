import { apiJson } from "./client";
import type { GenerateRequest, GenerationMemoryEstimate } from "./types";

export type EstimateFit = "fits" | "tight" | "wont-fit" | "unknown";

/**
 * VRAM preflight for a pending generation. Takes the full request; only the
 * model + dimensions materially move the estimate, but the server accepts the
 * whole shape.
 */
export function estimateGeneration(req: GenerateRequest): Promise<GenerationMemoryEstimate> {
  return apiJson<GenerationMemoryEstimate>("/api/generate/estimate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

/**
 * Bucket an estimate into a fit verdict for the badge. "Tight" is the top ~8%
 * of available memory — it fits, but barely. Unknown when the server can't see
 * available memory (remote engine without telemetry).
 */
export function classifyFit(est: GenerationMemoryEstimate): EstimateFit {
  const available = est.available_memory_bytes;
  if (est.fits_available_memory === false) return "wont-fit";
  if (available == null) return "unknown";
  if (est.peak_memory_bytes > available) return "wont-fit";
  if (est.peak_memory_bytes > available * 0.92) return "tight";
  return "fits";
}

const gb = (bytes: number) => `${(bytes / 1_000_000_000).toFixed(1)} GB`;

/**
 * Badge copy. Every string leads with "VRAM" so the reader knows what is
 * being estimated — the old bare "Fits · est. 2.3 GB" read as a mystery.
 */
export function estimateLabel(
  fit: EstimateFit,
  peakBytes: number,
  availableBytes: number | null,
): string {
  switch (fit) {
    case "fits":
      return availableBytes !== null
        ? `VRAM · fits — est. ${gb(peakBytes)} of ${gb(availableBytes)}`
        : `VRAM · est. ${gb(peakBytes)}`;
    case "tight":
      return "VRAM · tight — close other apps";
    case "wont-fit":
      return "VRAM · won't fit on this GPU";
    default:
      return `VRAM · est. ${gb(peakBytes)}`;
  }
}

/** Plain-language tooltip for the badge. */
export const ESTIMATE_TOOLTIP =
  "Preflight estimate of peak GPU memory (VRAM) this print needs, versus the " +
  "connected engine's available memory. Advisory — actual use varies.";
