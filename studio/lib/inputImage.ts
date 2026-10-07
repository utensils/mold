import { blobToBase64 } from "./base64";
import { imageDimensionsFromBase64 } from "./imageDimensions";

// Match PictureImport in MoldClient: 12 typed images fit the 32 MiB inline budget.
export const INPUT_IMAGE_MAX_BYTES = 2 * 1024 * 1024;
export const INPUT_IMAGE_MAX_AXIS = 4096;

/** Proportional resizing only; the renderer retains transparency and orientation. */
export async function boundInputImage(
  base64: string,
  byteCount: number,
  size: { width: number; height: number },
  render: (width: number, height: number) => Promise<string>,
): Promise<string> {
  if (
    byteCount <= INPUT_IMAGE_MAX_BYTES &&
    Math.max(size.width, size.height) <= INPUT_IMAGE_MAX_AXIS
  )
    return base64;
  let side = Math.min(Math.max(size.width, size.height), INPUT_IMAGE_MAX_AXIS);
  while (side >= 1) {
    const scale = side / Math.max(size.width, size.height);
    const result = await render(
      Math.max(1, Math.round(size.width * scale)),
      Math.max(1, Math.round(size.height * scale)),
    );
    const bytes =
      Math.floor((result.length * 3) / 4) -
      (result.endsWith("==") ? 2 : result.endsWith("=") ? 1 : 0);
    if (bytes <= INPUT_IMAGE_MAX_BYTES) return result;
    if (side === 1) break;
    side = Math.max(1, Math.floor(side * 0.75));
  }
  throw new Error("Could not resize this input image.");
}

/** Input-only encoder. Export/copy paths continue to use the lossless raw encoder. */
export async function inputImageBase64(blob: Blob): Promise<string> {
  const base64 = await blobToBase64(blob);
  const size = imageDimensionsFromBase64(base64, ["png", "jpeg", "webp"]);
  // Audio/video and other non-picture inputs are unchanged.
  if (!size) return base64;
  if (
    blob.size <= INPUT_IMAGE_MAX_BYTES &&
    Math.max(size.width, size.height) <= INPUT_IMAGE_MAX_AXIS
  )
    return base64;
  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const decoded = new Image();
    decoded.onload = () => resolve(decoded);
    decoded.onerror = () =>
      reject(new Error("Could not read this input image."));
    decoded.src = `data:${blob.type.startsWith("image/") ? blob.type : "image/png"};base64,${base64}`;
  });
  // Browser decoding applies EXIF; draw with its oriented dimensions.
  return boundInputImage(
    base64,
    blob.size,
    { width: image.naturalWidth, height: image.naturalHeight },
    async (width, height) => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Could not resize this input image.");
      context.drawImage(image, 0, 0, width, height);
      const result = canvas.toDataURL("image/png");
      if (!result.startsWith("data:image/png;base64,"))
        throw new Error("Could not encode this input image.");
      return result.slice(result.indexOf(",") + 1);
    },
  );
}

/** Metadata always describes the normalized bytes, never the original File MIME. */
export function inputImageFacts(
  base64: string,
  filename: string,
): { mimeType: string; filename: string } {
  const mimeType = base64.startsWith("iVBOR")
    ? "image/png"
    : base64.startsWith("/9j/")
      ? "image/jpeg"
      : base64.startsWith("UklGR")
        ? "image/webp"
        : null;
  if (!mimeType) return { mimeType: "application/octet-stream", filename };
  const extension =
    mimeType === "image/png"
      ? "png"
      : mimeType === "image/jpeg"
        ? "jpg"
        : "webp";
  const matches =
    mimeType === "image/jpeg"
      ? /\.jpe?g$/i.test(filename)
      : filename.toLowerCase().endsWith(`.${extension}`);
  return {
    mimeType,
    filename: matches
      ? filename
      : filename.replace(/\.[^.]+$/, "") + `.${extension}`,
  };
}

export async function normalizeInputImage(
  base64: string,
  filename: string,
): Promise<{ base64: string; filename: string; mimeType: string }> {
  const dimensions = imageDimensionsFromBase64(base64, ["png", "jpeg", "webp"]);
  const byteCount =
    Math.floor((base64.length * 3) / 4) -
    (base64.endsWith("==") ? 2 : base64.endsWith("=") ? 1 : 0);
  if (
    !dimensions ||
    (byteCount <= INPUT_IMAGE_MAX_BYTES &&
      Math.max(dimensions.width, dimensions.height) <= INPUT_IMAGE_MAX_AXIS)
  ) {
    return { base64, ...inputImageFacts(base64, filename) };
  }
  const bytes = Uint8Array.from(atob(base64), (character) =>
    character.charCodeAt(0),
  );
  const result = await inputImageBase64(
    new Blob([bytes], { type: inputImageFacts(base64, filename).mimeType }),
  );
  return { base64: result, ...inputImageFacts(result, filename) };
}
