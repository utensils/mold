import { imageDimensionsFromBase64 } from "./imageDimensions";
import type { SourceImageMode } from "./generationCapabilities";

export interface CanvasSourceImage {
  base64: string;
  width?: number | null;
  height?: number | null;
}
/** The rendered recipe decides which staged endpoints may steer its canvas. */
export function canvasSource(input: {
  mode: SourceImageMode;
  supportsEndFrame: boolean;
  source?: CanvasSourceImage | null | undefined;
  end?: CanvasSourceImage | null | undefined;
  h3?:
    | {
        firstFrame?: { data: string; width: number; height: number } | null;
        lastFrame?: { data: string; width: number; height: number } | null;
      }
    | null
    | undefined;
}): CanvasSourceImage | null {
  if (input.mode === "h3-boundaries") {
    const frame = input.h3?.firstFrame?.data
      ? input.h3.firstFrame
      : input.h3?.lastFrame?.data
        ? input.h3.lastFrame
        : null;
    return frame
      ? { base64: frame.data, width: frame.width, height: frame.height }
      : null;
  }
  if (input.source?.base64) return input.source;
  return input.supportsEndFrame && input.end?.base64 ? input.end : null;
}

export function canvasSourceDimensions(
  image: CanvasSourceImage | null,
): { width: number; height: number } | null {
  if (!image?.base64) return null;
  return image.width && image.height
    ? { width: image.width, height: image.height }
    : imageDimensionsFromBase64(image.base64);
}
