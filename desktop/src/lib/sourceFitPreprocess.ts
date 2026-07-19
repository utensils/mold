/**
 * Submit-path source-fit preprocessing — the desktop port of the web SPA's
 * `GeneratePage.vue` `preprocessSourceIfNeeded` + `fitStillSourceToRequest`
 * flow. Pure policy logic lives here with the canvas work behind an
 * injectable {@link SourceFitCanvasOps} boundary (happy-dom has no real 2D
 * canvas, and the tests want exact assertions anyway); the DOM-backed
 * implementation is `sourceFitCanvas.ts`.
 *
 * Flow (mirrors web):
 *   1. `upscale-then-fit` first rewrites the source through the upscaler —
 *      the caller supplies the upscale function already bound to the
 *      CONCRETE generation host so the upscaler model auto-downloads where
 *      the generation will run.
 *   2. If the (possibly upscaled) source already matches the target
 *      dimensions, nothing else happens.
 *   3. Otherwise the source is drawn onto a target-sized canvas per the
 *      policy, and a mask is (re)built when the policy is pad-repaint
 *      (white padding bands = repaint) or a user mask already exists (it
 *      must track the fitted source).
 */
import {
  maskPaddingRectangles,
  resolveSourceFitTransform,
  type Rect,
  type Size,
  type SourceFitPolicy,
  type SourceFitTransform,
} from "./sourceFit";

/** Canvas operations the preprocess needs — injectable for tests. */
export interface SourceFitCanvasOps {
  /** Decode a base64 image (no data-URI prefix) and report its pixel size. */
  imageSize(base64: string): Promise<Size>;
  /** Draw the image through `transform` onto a black target-sized canvas → base64 PNG. */
  fitImage(base64: string, transform: SourceFitTransform): Promise<string>;
  /**
   * Build the fitted mask: black canvas, the existing mask (if any) drawn
   * through `transform`, then `padRects` filled white (repaint). → base64 PNG.
   */
  buildMask(
    existingMask: string | null,
    transform: SourceFitTransform,
    padRects: Rect[],
  ): Promise<string>;
}

/** Runs the source through an upscaler model, returning the new base64 image. */
export type UpscaleFn = (image: string, model: string) => Promise<string>;

export interface SourceFitInput {
  /** img2img source, base64 (no data-URI prefix). */
  source: string | null;
  /** Inpaint mask, base64 — refit alongside the source when present. */
  mask: string | null;
  policy: SourceFitPolicy | undefined;
  target: Size;
}

export interface SourceFitResult {
  source: string | null;
  mask: string | null;
  /** True when either image was rewritten (upscaled and/or refit). */
  changed: boolean;
}

/**
 * The policy actually used for the canvas draw. A missing policy falls back
 * to pad-repaint (web parity); `upscale-then-fit` draws its inner `fit` —
 * the UI defaults that inner fit to pad-repaint for image families, and
 * maskless (video) img2img coerces it away so no unrepaintable pad bands
 * are ever drawn.
 */
export function drawableFitPolicy(policy: SourceFitPolicy | undefined): SourceFitPolicy {
  if (!policy) return { mode: "pad-repaint" };
  if (policy.mode === "upscale-then-fit") return policy.fit;
  return policy;
}

export async function applySourceFitPreprocess(
  input: SourceFitInput,
  deps: {
    ops: SourceFitCanvasOps;
    upscale?: UpscaleFn;
    onStatus?: (message: string) => void;
  },
): Promise<SourceFitResult> {
  let { source, mask } = input;
  let changed = false;
  if (!source) return { source, mask, changed };

  // 1. Upscaler preprocessing (upscale-then-fit only).
  const policy = input.policy;
  if (policy?.mode === "upscale-then-fit" && policy.upscalerModel && deps.upscale) {
    deps.onStatus?.(`Preprocessing source with ${policy.upscalerModel}`);
    source = await deps.upscale(source, policy.upscalerModel);
    changed = true;
  }

  // 2. Already the requested size — nothing to fit.
  const size = await deps.ops.imageSize(source);
  if (size.width === input.target.width && size.height === input.target.height) {
    return { source, mask, changed };
  }

  // 3. Canvas fit + mask generation.
  const fitPolicy = drawableFitPolicy(policy);
  const transform = resolveSourceFitTransform(size, input.target, fitPolicy);
  deps.onStatus?.(`Fitting source to ${input.target.width}×${input.target.height}`);
  source = await deps.ops.fitImage(source, transform);
  changed = true;
  if (fitPolicy.mode === "pad-repaint" || mask) {
    mask = await deps.ops.buildMask(mask, transform, maskPaddingRectangles(transform));
  }
  return { source, mask, changed };
}
