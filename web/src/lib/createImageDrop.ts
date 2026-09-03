/**
 * Where a dropped image lands on the web Create page.
 *
 * The POLICY is shared (`@studio/lib/imageDropRouting`); this module is the
 * web shell's half of it: it assembles the routing state from the form, and
 * writes the result into the SAME form fields the wells' own pickers write,
 * so a drag and a click produce identical facts. Extracted from
 * `CreatePage.vue` so both halves are unit-testable without mounting the page.
 */

import {
  resolveDropTarget,
  type DropRouting,
  type DropTarget,
} from "@studio/lib/imageDropRouting";
import type { SourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import { defaultSourceFitPolicy } from "@studio/lib/sourceFit";
import {
  appendMinimaxH3PickedImageReferences,
  MINIMAX_H3_MAX_REFERENCES,
  setMinimaxH3PickedImageBoundary,
} from "@studio/lib/minimaxH3Authoring";
import type { GenerateFormState, SourceImageState } from "../types";

export interface CreateDropContext {
  plan: SourceMediaPlan;
  /** The advertised strip ceiling; `null` is unbounded (Qwen edit). */
  referenceMax: number | null;
  /** The server's own sentence for a recipe that takes no references. */
  refusalReason: string | null;
  /** The identity-photo well is rendering (a qualified checkpoint). */
  identityVisible: boolean;
  /** The sequence Opening image well is rendering. */
  openingVisible: boolean;
}

/** How many references the strip holds, in whichever store this plan uses. */
export function droppedReferenceCount(
  state: GenerateFormState,
  plan: SourceMediaPlan,
): number {
  return plan.kind === "single-or-references"
    ? (state.referenceImages?.length ?? 0)
    : state.imageAttachments.length;
}

/** Route one dropped file with no well under the cursor (the window path). */
export function routeCreateDrop(
  state: GenerateFormState,
  context: CreateDropContext,
): DropRouting {
  return resolveDropTarget(context.plan, null, {
    hasSource: Boolean(state.imageAttachments[0]?.base64),
    referenceCount: droppedReferenceCount(state, context.plan),
    identityVisible: context.identityVisible,
    openingVisible: context.openingVisible,
    h3FirstPresent: Boolean(state.h3Authoring?.firstFrame),
    h3ReferenceCount: state.h3Authoring?.references.length ?? 0,
    h3ReferenceMax: MINIMAX_H3_MAX_REFERENCES,
    lastWrite: state.exclusiveWell ?? null,
    refusalReason: context.refusalReason,
  });
}

/** The sequence draft slot a drop can write. */
export interface OpeningImageDraft {
  openingImage: { filename: string; base64: string | null } | null;
}

/**
 * Write the dropped image into the form. Strips APPEND — the click path
 * appends too, and a drop that replaced the strip lost every earlier picture.
 * Returns an error sentence when the write itself was refused (H3 validates
 * its own media), otherwise `null`.
 */
export async function applyCreateDrop(
  state: GenerateFormState,
  target: DropTarget,
  image: SourceImageState,
  context: Pick<CreateDropContext, "plan" | "referenceMax">,
  draft?: OpeningImageDraft | null,
): Promise<string | null> {
  switch (target) {
    case "source":
      state.imageAttachments =
        context.plan.kind === "attachments"
          ? [image, ...state.imageAttachments.slice(1)]
          : [image];
      state.sourceFitPolicy = defaultSourceFitPolicy();
      // Last write wins on an exclusive recipe: the references park, kept.
      state.exclusiveWell = "source";
      return null;
    case "references": {
      const max = context.referenceMax ?? undefined;
      if (context.plan.kind === "single-or-references") {
        state.referenceImages = [...(state.referenceImages ?? []), image].slice(
          0,
          max,
        );
      } else {
        state.imageAttachments = [...state.imageAttachments, image].slice(
          0,
          max,
        );
      }
      state.exclusiveWell = "references";
      return null;
    }
    case "end":
      state.endFrame = image;
      return null;
    case "identity":
      state.identityImage = image;
      return null;
    case "opening":
      if (!draft) return "This sequence has no opening-image well.";
      draft.openingImage = { filename: image.filename, base64: image.base64 };
      return null;
    case "h3-first":
    case "h3-last": {
      const result = setMinimaxH3PickedImageBoundary(
        state.h3Authoring,
        target === "h3-first" ? "firstFrame" : "lastFrame",
        h3Source(image),
      );
      if (!result.ok) return result.error;
      state.h3Authoring = result.state;
      return null;
    }
    case "h3-reference": {
      const result = await appendMinimaxH3PickedImageReferences(
        state.h3Authoring,
        [h3Source(image)],
      );
      if (!result.ok) return result.error;
      state.h3Authoring = result.state;
      return null;
    }
  }
}

function h3Source(image: SourceImageState) {
  return {
    filename: image.filename,
    base64: image.base64,
    ...(image.width ? { width: image.width } : {}),
    ...(image.height ? { height: image.height } : {}),
    ...(image.mime ? { mimeType: image.mime } : {}),
  };
}
