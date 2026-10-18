import type { ComputedRef, InjectionKey } from "vue";
export type SourceAttachmentIntent = "first" | "last";
export interface SourceAttachmentCapabilities {
  supportsKeyframes: boolean;
  supportsEndFrame: boolean;
  supportsSourceImage: boolean;
  boundaryFrames?:
    import("./generated/generationProfileV1").BoundaryFramesProfile | null;
  sourceImageMode: string;
  requiresSourceImage: boolean;
}
export const SOURCE_ATTACHMENT_CHOICES: InjectionKey<
  ComputedRef<SourceAttachmentIntent[]>
> = Symbol("source-attachment-choices");
/** Already-resolved recipe capabilities are the sole authority; legacy fallback stays at their resolver. */
export function imageAttachmentChoices(
  caps: SourceAttachmentCapabilities,
): SourceAttachmentIntent[] {
  if (caps.boundaryFrames) {
    if (!["adjustable", "fixed"].includes(caps.boundaryFrames.mode)) return [];
    if (caps.boundaryFrames.wire !== "h3-endpoints") return [];
    return caps.boundaryFrames.first_required ? ["first"] : ["first", "last"];
  }
  if (caps.sourceImageMode === "h3-boundaries")
    return caps.requiresSourceImage ? ["first"] : ["first", "last"];
  if (caps.supportsEndFrame || caps.supportsKeyframes) return ["first", "last"];
  return [];
}
export function replaceEndpointKeyframe<
  T,
  F extends { frame: number; image: T },
>(
  frames: readonly F[],
  count: number,
  intent: SourceAttachmentIntent,
  image: T,
): F[] {
  if (!Number.isSafeInteger(count) || count < 2)
    throw new Error(
      "Choose a valid video frame count before attaching an endpoint.",
    );
  const frame = intent === "first" ? 0 : count - 1;
  const index = frames.findIndex((entry) => entry.frame === frame);
  if (index >= 0)
    return frames.map((entry, position) =>
      position === index ? { ...entry, image } : entry,
    );
  const result = [...frames];
  const at = result.findIndex((entry) => entry.frame > frame);
  result.splice(at < 0 ? result.length : at, 0, { frame, image } as F);
  return result;
}
