/**
 * DOM-backed {@link SourceFitCanvasOps} — the real canvas implementation the
 * Tauri webview uses (full HTML canvas support). Kept apart from the policy
 * logic in `sourceFitPreprocess.ts` so tests can inject a fake and stay off
 * happy-dom's stub canvas.
 */
import type { Rect, Size, SourceFitTransform } from "./sourceFit";
import type { SourceFitCanvasOps } from "./sourceFitPreprocess";
import { base64ToDataUrl } from "./image";

function loadImage(base64: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("failed to decode the source image"));
    img.src = base64ToDataUrl(base64);
  });
}

function makeCanvas(transform: SourceFitTransform): {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
} {
  const canvas = document.createElement("canvas");
  canvas.width = transform.outputWidth;
  canvas.height = transform.outputHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2D context unavailable");
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  return { canvas, ctx };
}

/** Strip the data-URI prefix — the wire wants bare base64. */
function canvasToBase64(canvas: HTMLCanvasElement): string {
  const dataUrl = canvas.toDataURL("image/png");
  const comma = dataUrl.indexOf(",");
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

export const domCanvasOps: SourceFitCanvasOps = {
  async imageSize(base64: string): Promise<Size> {
    const img = await loadImage(base64);
    return { width: img.naturalWidth, height: img.naturalHeight };
  },

  async fitImage(base64: string, transform: SourceFitTransform): Promise<string> {
    const img = await loadImage(base64);
    const { canvas, ctx } = makeCanvas(transform);
    ctx.drawImage(
      img,
      transform.offsetX,
      transform.offsetY,
      transform.drawWidth,
      transform.drawHeight,
    );
    return canvasToBase64(canvas);
  },

  async buildMask(
    existingMask: string | null,
    transform: SourceFitTransform,
    padRects: Rect[],
  ): Promise<string> {
    const { canvas, ctx } = makeCanvas(transform);
    if (existingMask) {
      const maskImg = await loadImage(existingMask);
      ctx.drawImage(
        maskImg,
        transform.offsetX,
        transform.offsetY,
        transform.drawWidth,
        transform.drawHeight,
      );
    }
    ctx.fillStyle = "white";
    for (const rect of padRects) {
      ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    }
    return canvasToBase64(canvas);
  },
};
