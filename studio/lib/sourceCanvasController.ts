import { canvasSourceDimensions, type CanvasSourceImage } from "./canvasSource";
import type { CanvasIntent } from "./outputShape";
import {
  resolveDefaultSourceResolution,
  resolveSourceCanvasTransition,
  resolveSourceResolution,
  type ModelResolutionContract,
  type SourceDimensions,
} from "./sourceResolution";

export interface SourceCanvasInput {
  source: CanvasSourceImage | null;
  closingSource: string;
  recipeKey: string;
  model: ModelResolutionContract | string;
  pipeline: string | null;
  intent: CanvasIntent;
  referenceOnly: boolean;
  primaryIsTarget: boolean;
  canvasless: boolean;
}

/** The authoring transition authority. Shells choose the active endpoint and
 * apply the returned projection; only this controller remembers whether that
 * endpoint was restored, replaced, or exposed by removing the opening frame. */
export class SourceCanvasController {
  private previousSource = "";
  private previousDimensions: SourceDimensions | null = null;
  private previousClosing = "";
  private previousRecipe = "";
  private initialized = false;
  private preservedReplacement = "";

  preserveRestored(base64: string): void {
    this.preservedReplacement = base64;
  }

  update(input: SourceCanvasInput): {
    dimensions: SourceDimensions | null;
    canvas: SourceDimensions | null;
    intent: CanvasIntent;
    replaced: boolean;
  } {
    const selected = input.source?.base64 ?? "";
    if (
      input.intent === "manual" &&
      selected &&
      (!this.initialized ||
        input.recipeKey !== this.previousRecipe ||
        (selected === input.closingSource &&
          this.previousSource &&
          input.closingSource === this.previousClosing))
    )
      this.preservedReplacement = selected;
    this.previousClosing = input.closingSource;
    this.previousRecipe = input.recipeKey;
    this.initialized = true;
    const replaced = Boolean(selected && selected !== this.previousSource);
    const dimensions =
      selected === this.previousSource && this.previousDimensions
        ? this.previousDimensions
        : canvasSourceDimensions(input.source);
    this.previousSource = selected;
    this.previousDimensions = dimensions;
    let intent = input.intent;
    if (!selected) {
      if (intent !== "manual") intent = "model-default";
      return { dimensions: null, canvas: null, intent, replaced };
    }
    if (!dimensions || (input.referenceOnly && !input.primaryIsTarget)) {
      return { dimensions, canvas: null, intent, replaced };
    }
    const preserveReplacement =
      replaced && this.preservedReplacement === selected;
    const canvas = resolveSourceCanvasTransition({
      source: resolveSourceResolution(dimensions, input.model, input.pipeline),
      automatic: resolveDefaultSourceResolution(
        dimensions,
        input.model,
        input.pipeline,
      ),
      replaced,
      intent,
      preserveReplacement,
    });
    if (replaced) {
      this.preservedReplacement = "";
      if (preserveReplacement) intent = "manual";
      else if (intent !== "source-exact") intent = "source";
    }
    return {
      dimensions,
      canvas: input.canvasless ? null : canvas,
      intent,
      replaced,
    };
  }
}
