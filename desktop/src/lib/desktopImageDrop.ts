import { generationCapabilitiesForFamily } from "./capabilities";
import type { ModelEntry, OutputMetadata } from "./api/types";
import { applyMetadataToForm, type GenerateForm, type PickedImage } from "./generateForm";
import { defaultSourceFitPolicy } from "@studio/lib/sourceFit";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import { sourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import { resolveDropTarget, type DropTarget } from "@studio/lib/imageDropRouting";
import {
  appendMinimaxH3PickedImageReferences,
  emptyMinimaxH3AuthoringState,
  minimaxH3ReferenceBudget,
  MINIMAX_H3_MAX_REFERENCES,
  setMinimaxH3PickedImageBoundary,
} from "@studio/lib/minimaxH3Authoring";

/** A local still read by the native desktop backend after an OS file drop. */
export interface DesktopImageImport {
  filename: string;
  base64: string;
  width: number;
  height: number;
  /** Content key used by the native app to reopen the original local path. */
  sha256?: string;
  metadata: OutputMetadata | null;
}

export interface DesktopImageDropResult {
  attached: boolean;
  metadataApplied: boolean;
  /** Which well received the image, when one did. */
  target?: DropTarget;
  /** Why nothing was attached — the sentence to show the user. */
  refused?: string;
}

/**
 * What the Create route is rendering beside the source wells. The identity
 * photo is its own well, orthogonal to the source-media plan, so the view
 * tells the router whether it exists.
 */
export interface DesktopDropContext {
  identityVisible?: boolean;
}

/**
 * Apply one native file drop to Generate.
 *
 * Tauri's `dragDropEnabled` swallows a Finder drag before any HTML5 `drop`
 * fires, so this — reached from ONE window-level bridge — is the only handler
 * an OS drop ever gets. It used to route by MODEL CAPABILITY, which replaced
 * the whole attachment strip on every drop, wrote `imageAttachments` for H3
 * (a field its request builder never reads, so the drop visibly did nothing),
 * and left the identity, end-frame and H3 wells unreachable
 * by any drag at all. It now asks the SHARED router where the drop belongs —
 * the well under the cursor when the plan renders one, the plan's default
 * otherwise — and every strip APPENDS.
 *
 * Embedded Mold metadata still restores the complete serialized generation
 * first; the routing then runs against the model that restore selected.
 */
export async function applyDesktopImageDrop(
  form: GenerateForm,
  image: DesktopImageImport,
  models: ModelEntry[] = [],
  hovered: DropTarget | null = null,
  context: DesktopDropContext = {},
): Promise<DesktopImageDropResult> {
  const metadataApplied = image.metadata != null;
  if (image.metadata) applyMetadataToForm(form, image.metadata, models);

  // The selected row is the authority on the checkpoint's contract — the
  // reference block that decides Klein's two wells rides its advertised
  // recipe, exactly as `SourceImageWell` reads it.
  const entry = models.find((model) => model.name === form.model) ?? null;
  const caps = generationCapabilitiesForFamily(
    form.family,
    form.model,
    form.pipeline,
    entry?.guidance_capabilities ?? form.guidanceCapabilities,
    entry?.source_image ?? form.sourceImageCapability,
    effectiveGenerationRecipe(entry, form.pipeline),
  );
  const plan = sourceMediaPlan(caps);
  const h3References = form.h3Authoring?.references ?? [];
  const routed = resolveDropTarget(plan, hovered, {
    hasSource: Boolean(form.sourceImage),
    referenceCount: form.imageAttachments.length,
    identityVisible: context.identityVisible === true,
    // Scene-by-scene authoring is retired, so there is no opening-image well
    // for a drop to land in any more.
    openingVisible: false,
    h3FirstPresent: Boolean(form.h3Authoring?.firstFrame),
    h3ReferenceCount: h3References.length,
    h3ReferenceMax: MINIMAX_H3_MAX_REFERENCES,
    lastWrite: form.exclusiveWell ?? null,
    // The refusal is the SERVER's sentence when it wrote one.
    refusalReason: caps.referenceImagesReason,
  });
  if (typeof routed !== "string") {
    return { attached: false, metadataApplied, refused: routed.refused };
  }

  const applied = await applyDropToForm(form, routed, image);
  return applied.ok
    ? { attached: true, metadataApplied, target: routed }
    : { attached: false, metadataApplied, refused: applied.error };
}

/**
 * Write the dropped image into the SAME store fields the well's own `file`
 * emit writes, so a drag and a click produce identical facts.
 */
export async function applyDropToForm(
  form: GenerateForm,
  target: DropTarget,
  image: DesktopImageImport,
): Promise<{ ok: true } | { ok: false; error: string }> {
  const picked: PickedImage = { filename: image.filename, base64: image.base64 };
  switch (target) {
    case "source":
      form.sourceImage = image.base64;
      form.sourceImageName = image.filename;
      form.sourceFit = defaultSourceFitPolicy();
      form.sourceImageWidth = image.width;
      form.sourceImageHeight = image.height;
      // Klein's two wells are mutually exclusive: this write parks the
      // references without discarding them.
      form.exclusiveWell = "source";
      return { ok: true };
    case "references": {
      // APPEND. The click-to-add path appends, and a drop that replaced the
      // strip lost a Target and two references on the third drop.
      const establishesTarget = form.imageAttachments.length === 0;
      form.imageAttachments = [...form.imageAttachments, image.base64];
      if (establishesTarget) {
        form.sourceFit = defaultSourceFitPolicy();
        form.sourceImageWidth = image.width;
        form.sourceImageHeight = image.height;
      }
      form.exclusiveWell = "references";
      return { ok: true };
    }
    case "end":
      form.endFrame = picked;
      return { ok: true };
    case "identity":
      form.identityImage = picked;
      return { ok: true };
    case "h3-first":
    case "h3-last": {
      const result = setMinimaxH3PickedImageBoundary(
        form.h3Authoring ?? emptyMinimaxH3AuthoringState(),
        target === "h3-first" ? "firstFrame" : "lastFrame",
        {
          filename: image.filename,
          base64: image.base64,
          width: image.width,
          height: image.height,
        },
      );
      if (!result.ok) return { ok: false, error: result.error };
      form.h3Authoring = result.state;
      return { ok: true };
    }
    case "h3-reference": {
      const result = await appendMinimaxH3PickedImageReferences(
        form.h3Authoring ?? emptyMinimaxH3AuthoringState(),
        [
          {
            filename: image.filename,
            base64: image.base64,
            width: image.width,
            height: image.height,
          },
        ],
      );
      if (!result.ok) return { ok: false, error: result.error };
      form.h3Authoring = result.state;
      const budget = minimaxH3ReferenceBudget(result.state.references);
      const error = budget.errors[0];
      if (error) {
        // The panel's own budget refusal, applied to the drop rather than
        // left for Generate to discover.
        form.h3Authoring = {
          ...result.state,
          references: result.state.references.slice(0, -1),
        };
        return { ok: false, error };
      }
      return { ok: true };
    }
    default:
      // `resolveDropTarget` is never asked for a well this surface does not
      // render, so reaching here means a routing target with no home.
      return { ok: false, error: "That image has nowhere to go on this style." };
  }
}
