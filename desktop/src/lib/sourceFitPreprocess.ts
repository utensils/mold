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
import type { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import {
  coerceSourceFitForMaskless,
  maskPaddingRectangles,
  resolveSourceFitTransform,
  type Rect,
  type Size,
  type SourceFitPolicy,
  type SourceFitTransform,
} from "@studio/lib/sourceFit";
import type { MinimaxH3AuthoringState } from "@studio/lib/minimaxH3Authoring";

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
    cache?: SourceFitPreprocessCache;
    onStatus?: (message: string) => void;
  },
): Promise<SourceFitResult> {
  let { source, mask } = input;
  let changed = false;
  if (!source) return { source, mask, changed };

  // 1. Upscaler preprocessing (upscale-then-fit only). This cache layer is
  // dimension-independent, so changing resolution retains the expensive result.
  const policy = input.policy;
  if (policy?.mode === "upscale-then-fit" && policy.upscalerModel && deps.upscale) {
    const runUpscale = () => {
      deps.onStatus?.(`Preprocessing source with ${policy.upscalerModel}`);
      return deps.upscale!(source!, policy.upscalerModel);
    };
    source = deps.cache
      ? await deps.cache.upscale(source, policy.upscalerModel, runUpscale)
      : await runUpscale();
    changed = true;
  }

  // 2–3. Cache the size check, canvas fit, and mask generation together.
  const fitPolicy = drawableFitPolicy(policy);
  const fitSource = source;
  const fitMask = mask;
  const runFit = async (): Promise<SourceFitResult> => {
    const size = await deps.ops.imageSize(fitSource);
    if (size.width === input.target.width && size.height === input.target.height) {
      return { source: fitSource, mask: fitMask, changed };
    }

    const transform = resolveSourceFitTransform(size, input.target, fitPolicy);
    deps.onStatus?.(`Fitting source to ${input.target.width}×${input.target.height}`);
    const fittedSource = await deps.ops.fitImage(fitSource, transform);
    let fittedMask = fitMask;
    if (fitPolicy.mode === "pad-repaint" || fitMask) {
      fittedMask = await deps.ops.buildMask(fitMask, transform, maskPaddingRectangles(transform));
    }
    return { source: fittedSource, mask: fittedMask, changed: true };
  };

  return deps.cache
    ? deps.cache.fit(fitSource, fitMask, fitPolicy, input.target, runFit)
    : runFit();
}

/**
 * Fit both MiniMax H3 FL2VA boundaries onto the render canvas with the same
 * client-side machinery as an ordinary source image. H3 has no repaint mask,
 * so the policy is coerced maskless; an untouched boundary (already at the
 * target size) keeps its provenance, while a rewritten one drops its stale
 * sha256 and records the new dimensions.
 */
export async function applyH3BoundaryFit(
  state: MinimaxH3AuthoringState | null | undefined,
  policy: SourceFitPolicy,
  target: { width: number; height: number },
  deps: Parameters<typeof applySourceFitPreprocess>[1],
): Promise<MinimaxH3AuthoringState | null> {
  if (!state || (!state.firstFrame?.data && !state.lastFrame?.data)) {
    return state ?? null;
  }
  const maskless = coerceSourceFitForMaskless(policy);
  const next = { ...state };
  for (const endpoint of ["firstFrame", "lastFrame"] as const) {
    const boundary = next[endpoint];
    if (!boundary?.data) continue;
    const result = await applySourceFitPreprocess(
      { source: boundary.data, mask: null, policy: maskless, target },
      deps,
    );
    if (!result.changed || !result.source) continue;
    next[endpoint] = {
      ...boundary,
      data: result.source,
      mimeType: "image/png",
      width: target.width,
      height: target.height,
      sha256: null,
    };
  }
  return next;
}
