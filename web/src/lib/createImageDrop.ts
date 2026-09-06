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

/**
 * Where the drop actually LANDED, as its `data-drop-target` value.
 *
 * The web mirror of desktop's `dropTargetAtPosition`: the same attribute, the
 * same `closest(…)` walk from the hit-tested element, so a hit on a thumbnail
 * inside a well resolves to the well. A `DragEvent`'s `clientX`/`clientY` are
 * already CSS pixels, so unlike the Tauri bridge there is no
 * `devicePixelRatio` division. `null` means the drop landed on chrome, which
 * the shared router reads as "use the plan default".
 */
export function dropTargetAtPoint(
  point: DropPoint | null | undefined,
): DropTarget | null {
  if (!point || typeof document === "undefined") return null;
  const element = document.elementFromPoint(point.clientX, point.clientY);
  const well = element?.closest("[data-drop-target]");
  return (well?.getAttribute("data-drop-target") as DropTarget | null) ?? null;
}

/** The two coordinates a `DragEvent` carries; nothing else is read. */
export interface DropPoint {
  clientX: number;
  clientY: number;
}

/**
 * Route one dropped file.
 *
 * `point` is the drop's own coordinates: the well under it wins whenever the
 * plan renders it, which is what makes a labelled strip a real drop target
 * even when the strip itself did not handle the event. Omit it (or drop on
 * chrome) and the plan's default receives the file.
 */
export function routeCreateDrop(
  state: GenerateFormState,
  context: CreateDropContext,
  point?: DropPoint | null,
): DropRouting {
  return resolveDropTarget(context.plan, dropTargetAtPoint(point), {
    hasSource: Boolean(state.imageAttachments[0]?.base64),
    referenceCount: droppedReferenceCount(state, context.plan),
    identityVisible: context.identityVisible,
    // The sequence Opening image well is retired on web, so the shared router
    // can never select `opening` here.
    openingVisible: false,
    h3FirstPresent: Boolean(state.h3Authoring?.firstFrame),
    h3ReferenceCount: state.h3Authoring?.references.length ?? 0,
    h3ReferenceMax: MINIMAX_H3_MAX_REFERENCES,
    lastWrite: state.exclusiveWell ?? null,
    refusalReason: context.refusalReason,
  });
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
      // Unreachable: `routeCreateDrop` always reports the well as absent. The
      // arm stays so the switch remains exhaustive over the shared target set.
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
