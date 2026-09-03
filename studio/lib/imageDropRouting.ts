/**
 * Where a dropped image FILE lands — one policy for every surface.
 *
 * The bug this exists to kill: on desktop, Tauri's `dragDropEnabled` swallows
 * the OS drag before any HTML5 `drop` fires, so a single window-level bridge
 * is the only handler that ever sees a Finder drop. That bridge used to route
 * by MODEL CAPABILITY — never by the well under the cursor — which replaced a
 * whole attachment strip on a two-reference drop, wrote a field the H3 request
 * builder does not read, and left the identity, sequence-opening, end-frame
 * and per-reference wells unreachable by any drag at all. On web the opposite
 * hole: no window-level handler, so a file dropped a pixel outside a well
 * navigated the browser to the image and took the SPA with it.
 *
 * The answer is one pure decision both shells call: the hovered well wins
 * when the plan renders it, otherwise the plan's own default receives the
 * drop, and a strip at its advertised ceiling refuses instead of silently
 * dropping the file. Every strip target APPENDS; nothing here ever replaces.
 */

import {
  resolveExclusiveWells,
  type ExclusiveWell,
  type SourceMediaPlan,
} from "./sourceMediaPlan";

/**
 * Every well a drop can land on. These are also the `data-drop-target` values
 * the components render, so `document.elementFromPoint(...)` can name one.
 */
export type DropTarget =
  | "source"
  | "end"
  | "references"
  | "identity"
  | "opening"
  | "h3-first"
  | "h3-last"
  | "h3-reference";

export interface DropRoutingState {
  /** The single source well (or Qwen's Target) holds media. */
  hasSource: boolean;
  /** How many ordered references the strip already holds. */
  referenceCount: number;
  /** The identity-photo well is rendering (a qualified checkpoint). */
  identityVisible: boolean;
  /** The sequence Opening image well is rendering. */
  openingVisible: boolean;
  /** H3's first-frame boundary is filled, so the default moves to the last. */
  h3FirstPresent: boolean;
  h3ReferenceCount?: number;
  h3ReferenceMax?: number | null;
  /** Which exclusive well the user wrote last (`single-or-references`). */
  lastWrite?: ExclusiveWell | null;
  /** The profile's own sentence for a model that takes no image input. */
  refusalReason?: string | null;
}

export type DropRouting = DropTarget | { refused: string };

/** The fallback sentence when the recipe supplied none. */
export const NO_IMAGE_INPUT_REFUSAL =
  "This model doesn't accept a source image.";

/** The count sentence, mirroring the server's own reference-count refusal. */
export function referenceCountRefusal(max: number): string {
  return `This model supports at most ${max} reference image${max === 1 ? "" : "s"}.`;
}

/** The reference strip's ceiling for this plan; `null` is unbounded. */
function referenceMax(plan: SourceMediaPlan): number | null {
  if (plan.kind === "attachments") return plan.max;
  if (plan.kind === "single-or-references") return plan.references.max;
  return null;
}

/** Whether this plan renders the named well at all. */
function planRenders(
  plan: SourceMediaPlan,
  target: DropTarget,
  state: DropRoutingState,
): boolean {
  // The identity photo and the sequence opening image are their own wells,
  // orthogonal to the source-media plan — the surface says whether they show.
  if (target === "identity") return state.identityVisible;
  if (target === "opening") return state.openingVisible;
  switch (plan.kind) {
    case "none":
      return false;
    case "single":
      return target === "source" || (target === "end" && plan.endFrame);
    case "attachments":
      // Qwen's Target is the shared source well above its strip.
      return (
        target === "references" ||
        (target === "source" && plan.primary === "target")
      );
    case "single-or-references":
      return (
        target === "source" ||
        target === "references" ||
        (target === "end" && plan.single.endFrame)
      );
    case "h3-boundaries":
      return target === "h3-first" || target === "h3-last";
    case "h3-references":
      return target === "h3-reference";
  }
}

/** The well a drop lands on when nothing is under the cursor. */
function planDefault(
  plan: SourceMediaPlan,
  state: DropRoutingState,
): DropTarget | null {
  switch (plan.kind) {
    case "none":
      return null;
    case "single":
      return "source";
    case "attachments":
      return "references";
    case "single-or-references":
      // The well that is already active keeps receiving; an empty form starts
      // with the source, which is the img2img default Klein ships with.
      return (
        resolveExclusiveWells({
          hasSource: state.hasSource,
          referenceCount: state.referenceCount,
          lastWrite: state.lastWrite ?? null,
        }).active ?? "source"
      );
    case "h3-boundaries":
      return state.h3FirstPresent ? "h3-last" : "h3-first";
    case "h3-references":
      return "h3-reference";
  }
}

/** A strip target that is already full refuses rather than losing the file. */
function stripRefusal(
  plan: SourceMediaPlan,
  target: DropTarget,
  state: DropRoutingState,
): string | null {
  if (target === "references") {
    const max = referenceMax(plan);
    if (max !== null && state.referenceCount >= max) {
      return referenceCountRefusal(max);
    }
    return null;
  }
  if (target === "h3-reference") {
    const max = state.h3ReferenceMax ?? null;
    if (max !== null && (state.h3ReferenceCount ?? 0) >= max) {
      return referenceCountRefusal(max);
    }
  }
  return null;
}

/**
 * Route one dropped image.
 *
 * `hovered` is the `data-drop-target` under the cursor (the shells resolve it
 * from the drop coordinates), or `null` when the drop landed on chrome. A
 * hovered well the plan does not render falls back to the plan default rather
 * than writing state nothing can show.
 */
export function resolveDropTarget(
  plan: SourceMediaPlan,
  hovered: DropTarget | null,
  state: DropRoutingState,
): DropRouting {
  const target =
    hovered && planRenders(plan, hovered, state)
      ? hovered
      : planDefault(plan, state);
  if (!target) {
    return { refused: state.refusalReason || NO_IMAGE_INPUT_REFUSAL };
  }
  const refused = stripRefusal(plan, target, state);
  return refused ? { refused } : target;
}
