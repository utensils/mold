import type { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import {
  effectiveGenerationRecipe,
  fixedRecipeControlOverrides,
} from "@studio/lib/generationProfile";
import { resolveSourceConditioningTarget } from "@studio/lib/sourceResolution";
import { coerceSourceFitForMaskless } from "@studio/lib/sourceFit";
import { conditioningForRequest } from "@studio/lib/sourceMediaPlan";
import { isMeshFamily } from "@studio/lib/legacyRecipeRules";
import {
  applyMinimaxH3ReferenceCrops,
  emptyMinimaxH3AuthoringState,
} from "@studio/lib/minimaxH3Authoring";
import type { ApiTarget } from "../lib/api/client";
import type { GenerateRequest, ModelEntry } from "../lib/api/types";
import { generationCapabilitiesForForm } from "../lib/capabilities";
import { buildRequest, type GenerateForm } from "../lib/generateForm";
import { mobileMediaBudgetValidationError } from "../lib/generateValidation";
import {
  applyH3BoundaryFit,
  applySourceFitPreprocess,
  type SourceFitCanvasOps,
} from "../lib/sourceFitPreprocess";

export interface PrepareMobileGenerationInput {
  target: ApiTarget;
  /** Private frozen clone; preparation may rewrite its media in place. */
  draft: GenerateForm;
  selectedModel: ModelEntry | null;
  isCurrent?: () => boolean;
  signal?: AbortSignal;
}

export interface MobileGenerationPreparationServices {
  cache: SourceFitPreprocessCache;
  ops: SourceFitCanvasOps;
  upscale(
    image: string,
    model: string,
    target: ApiTarget,
    signal: AbortSignal | undefined,
    onProgress: (message: string) => void,
  ): Promise<string>;
  onStatus(message: string): void;
}

/**
 * Convert one frozen mobile form into its final wire request.
 *
 * This function deliberately has no Vue or DOM access. The caller owns live
 * state and supplies canvas/network ports; every decision here is made from
 * the frozen draft and selected-model snapshot captured at the tap boundary.
 */
export async function prepareMobileGenerationRequest(
  input: PrepareMobileGenerationInput,
  services: MobileGenerationPreparationServices,
): Promise<GenerateRequest> {
  const isCurrent = input.isCurrent ?? (() => true);
  const { draft } = input;
  const report = (message: string) => {
    if (isCurrent()) services.onStatus(message);
  };
  const upscale = (image: string, model: string) =>
    services.upscale(image, model, input.target, input.signal, report);

  Object.assign(
    draft,
    fixedRecipeControlOverrides(effectiveGenerationRecipe(input.selectedModel, draft.pipeline)),
  );
  // The draft's own recipe snapshot decides the reference contract — the same
  // authority the wells rendered from and the same one `buildRequest` submits
  // under, so an exclusive (Klein) recipe never fits a PARKED source.
  const capabilities = generationCapabilitiesForForm(
    draft.family,
    draft.model,
    draft.pipeline,
    draft.guidanceCapabilities,
    draft.sourceImageCapability,
    draft.recipeCapabilities,
  );
  // The advertised recipe answers whether this render has a pixel canvas at
  // all; the pre-profile mesh family rule is the fallback for a host that
  // predates the field — exactly the pairing `buildRequest` uses.
  const canvasless = draft.recipeCapabilities?.canvasless ?? isMeshFamily(draft.family);
  const preprocessing = {
    ops: services.ops,
    cache: services.cache,
    upscale,
    onStatus: report,
  };

  if (capabilities.sourceImageMode === "h3-boundaries") {
    draft.h3Authoring =
      (await applyH3BoundaryFit(
        draft.h3Authoring,
        draft.sourceFit,
        { width: draft.width, height: draft.height },
        preprocessing,
      )) ?? emptyMinimaxH3AuthoringState();
  } else if (capabilities.sourceImageMode === "ordered-references" && draft.h3Authoring) {
    // Ref2VA: a pending image crop is applied at the original resolution
    // before the request is built, so upload conversion, placement preview,
    // and the frozen route all see the cropped reference.
    draft.h3Authoring = await applyMinimaxH3ReferenceCrops(draft.h3Authoring, services.ops);
  } else if (capabilities.sourceImageMode === "qwen-edit" && draft.imageAttachments[0]) {
    const target = resolveSourceConditioningTarget(
      { width: draft.width, height: draft.height },
      input.selectedModel ?? draft.family,
      draft.pipeline,
    );
    const result = await applySourceFitPreprocess(
      {
        source: draft.imageAttachments[0],
        mask: null,
        policy: coerceSourceFitForMaskless(draft.sourceFit),
        target,
      },
      preprocessing,
    );
    if (result.source) draft.imageAttachments[0] = result.source;
  } else if (
    capabilities.supportsImg2img &&
    // An exclusive (Klein) recipe fits the source only while it is the well
    // the request will actually carry; a parked source is never preprocessed.
    conditioningForRequest(capabilities.sourceImageMode, {
      hasSource: Boolean(draft.sourceImage),
      referenceCount: draft.imageAttachments.length,
      lastWrite: draft.exclusiveWell ?? null,
    }) === "source" &&
    draft.sourceImage &&
    // A canvasless recipe (a 3-D mesh) renders from the photo itself and
    // advertises a 0×0 canvas, so there is no target to fit toward — running
    // the ordinary fit would resize the conditioning image to nothing.
    // `buildRequest` records no `source_fit` for such a request either.
    !canvasless
  ) {
    const result = await applySourceFitPreprocess(
      {
        source: draft.sourceImage,
        mask: capabilities.supportsMask ? draft.maskImage : null,
        policy: capabilities.supportsMask
          ? draft.sourceFit
          : coerceSourceFitForMaskless(draft.sourceFit),
        target: { width: draft.width, height: draft.height },
      },
      preprocessing,
    );
    draft.sourceImage = result.source;
    draft.maskImage = result.mask;
  }

  const mediaBudgetError = mobileMediaBudgetValidationError(draft);
  if (mediaBudgetError) throw new Error(mediaBudgetError);
  return buildRequest(draft);
}
